// file: chess_gpt_full_game.dart

import 'dart:math';
import 'package:bishop/bishop.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';
import 'transformer_encoder.dart';

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      if (p.grad != 0.0) {
        p.data -= learningRate * p.grad;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

class ChessGptFullGame {
  final TransformerDecoder decoder;
  final TransformerEncoder encoder;
  final SGD optimizer;

  final Map<String, int> stoi;
  final Map<int, String> itos;
  final Map<String, int> boardStoi;
  final Map<int, String> boardItos;

  final int startTokenId;
  final int endTokenId;
  final int padTokenId;
  final int blockSize;

  ChessGptFullGame({
    required this.decoder,
    required this.encoder,
    required this.stoi,
    required this.itos,
    required this.boardStoi,
    required this.boardItos,
  })  : optimizer = SGD(decoder.parameters() + encoder.parameters(), 0.01),
        startTokenId = stoi['<start>']!,
        endTokenId = stoi['<end>']!,
        padTokenId = stoi['<pad>']!,
        blockSize = decoder.blockSize;

  // CORRECTED: The factory now receives the vocabularies as arguments
  static ChessGptFullGame create({
    int? projK,
    required (Map<String, int>, Map<int, String>) moveVocab,
    required (Map<String, int>, Map<int, String>) boardVocab,
  }) {
    // Unpack the provided vocabularies
    final (moveStoi, moveItos) = moveVocab;
    final (boardStoi, boardItos) = boardVocab;

    const int embedSize = 64;
    const int blockSize = 16;
    const int numLayers = 2;
    const int numHeads = 2;

    // Calculate vocabulary sizes from the provided maps
    final int boardVocabSize = boardStoi.length;
    final int moveVocabSize = moveStoi.length;

    final encoder = TransformerEncoder(
      vocabSize: boardVocabSize,
      embedSize: embedSize,
      blockSize: 64,
      numLayers: numLayers,
      numHeads: numHeads,
      projK: projK,
    );

    final decoder = TransformerDecoder(
      vocabSize: moveVocabSize,
      embedSize: embedSize,
      blockSize: blockSize,
      numLayers: numLayers,
      numHeads: numHeads,
      encoderEmbedSize: embedSize,
      projK: projK,
    );

    return ChessGptFullGame(
      decoder: decoder,
      encoder: encoder,
      stoi: moveStoi,
      itos: moveItos,
      boardStoi: boardStoi,
      boardItos: boardItos,
    );
  }

  static (Map<String, int>, Map<int, String>) _generateChessVocab(
      List<List<String>> gameSequences) {
    // This is no longer needed since the vocab is passed in.
    // However, keeping it for now in case you still need it.
    final Set<String> allMoves = {};
    for (final sequence in gameSequences) {
      allMoves.addAll(sequence);
    }
    final game = Game();
    void generateAllMoves(Game currentPosition, int depth) {
      if (depth == 0) return;
      final legalMoves = currentPosition.generateLegalMoves();
      for (final move in legalMoves) {
        allMoves.add(currentPosition.toAlgebraic(move));
        currentPosition.makeMove(move);
        generateAllMoves(currentPosition, depth - 1);
        currentPosition.undo();
      }
    }

    generateAllMoves(game, 2);
    final List<String> sortedUniqueMoves = allMoves.toList()..sort();
    final Map<String, int> stoi = {
      '<pad>': 0,
      '<start>': 1,
      '<end>': 2,
    };
    int counter = 3;
    for (final move in sortedUniqueMoves) {
      stoi[move] = counter++;
    }
    final Map<int, String> itos =
        stoi.map((key, value) => MapEntry(value, key));
    return (stoi, itos);
  }

  static (Map<String, int>, Map<int, String>) _createBoardVocab() {
    // This is no longer needed since the vocab is passed in.
    final Map<String, int> stoi = {};
    int idCounter = 0;
    stoi['empty'] = idCounter++;
    for (var color in ['W', 'B']) {
      for (var piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']) {
        stoi[piece + color] = idCounter++;
      }
    }
    final Map<int, String> itos =
        stoi.map((key, value) => MapEntry(value, key));
    return (stoi, itos);
  }

  static final Map<int, String> pieceTypeNames = {
    1: 'pawn',
    2: 'knight',
    3: 'bishop',
    4: 'rook',
    5: 'queen',
    6: 'king',
  };

  List<int> _tokenizeBoardStateFromFen(String fen, Map<String, int> boardStoi) {
    final game = Game();
    game.setup(fen: fen);

    final List<int> tokens = [];
    final board = game.board;
    final size = game.size;

    for (int i = 0; i < size.numIndices; i++) {
      if (!size.onBoard(i)) continue;

      final squareValue = board[i];

      if (squareValue.isEmpty) {
        tokens.add(boardStoi['empty']!);
      } else {
        final colourName = squareValue.colour == Bishop.white ? 'W' : 'B';
        final typeName = pieceTypeNames[squareValue.type]!;
        final pieceString = typeName + colourName;
        tokens.add(boardStoi[pieceString]!);
      }
    }
    return tokens;
  }

  Future<void> train(List<String> gameMoves) async {
    final game = Game();
    final List<int> tokenizedSeq = [startTokenId];
    for (final move in gameMoves) {
      tokenizedSeq.add(stoi[move]!);
      game.makeMove(game.getMove(move)!);
    }
    tokenizedSeq.add(endTokenId);

    while (tokenizedSeq.length < blockSize + 1) {
      tokenizedSeq.insert(0, padTokenId);
    }

    final List<int> inputSequence = tokenizedSeq.sublist(
        tokenizedSeq.length - blockSize - 1, tokenizedSeq.length - 1);
    final List<int> targetSequence = tokenizedSeq.sublist(
        tokenizedSeq.length - blockSize, tokenizedSeq.length);

    optimizer.zeroGrad();

    final boardStateTokens = _tokenizeBoardStateFromFen(game.fen, boardStoi);
    final encoderOutput = encoder.forward(boardStateTokens);

    final List<ValueVector> logits =
        decoder.forward(inputSequence, encoderOutput).$1;
    Value batchLoss = Value(0.0);
    int activeTokens = 0;
    for (int t = 0; t < logits.length; t++) {
      if (targetSequence[t] != padTokenId) {
        final ValueVector tokenLogits = logits[t];
        final int trueTargetId = targetSequence[t];
        final Value trueLogit = tokenLogits.values[trueTargetId];
        final Value sumExpLogits =
            tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
        final Value logSumExp = sumExpLogits.log();
        final Value negLogProb = logSumExp - trueLogit;
        batchLoss += negLogProb;
        activeTokens++;
      }
    }

    if (activeTokens > 0) {
      batchLoss = batchLoss / Value(activeTokens.toDouble());
    } else {
      batchLoss = Value(0.0);
    }
    batchLoss.backward();
    optimizer.step();
  }

  Future<String> predictNextMove(Game game) async {
    final List<String> gameHistoryAlgebraic = game.moveHistoryAlgebraic;
    final List<int> gameHistoryTokens =
        gameHistoryAlgebraic.map((m) => stoi[m]!).toList();
    final List<int> inputSequence = [startTokenId, ...gameHistoryTokens];

    if (inputSequence.length > blockSize) {
      inputSequence.removeRange(0, inputSequence.length - blockSize);
    }

    while (inputSequence.length < blockSize) {
      inputSequence.insert(0, padTokenId);
    }

    final boardStateTokens = _tokenizeBoardStateFromFen(game.fen, boardStoi);
    final encoderOutput = encoder.forward(boardStateTokens);

    final List<ValueVector> logits =
        decoder.forward(inputSequence, encoderOutput).$1;
    final ValueVector lastTokenLogits = logits.last;

    final List<Move> legalMoves = game.generateLegalMoves();
    final List<int> legalMoveIds =
        legalMoves.map((m) => stoi[game.toAlgebraic(m)]!).toList();

    for (int i = 0; i < lastTokenLogits.values.length; i++) {
      if (!legalMoveIds.contains(i)) {
        lastTokenLogits.values[i].data = -double.infinity;
      }
    }

    final ValueVector probabilities = lastTokenLogits.softmax();

    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    return itos[predictedNextToken]!;
  }
}
