// file: chess_gpt_full_game.dart

import 'dart:math';
import 'package:bishop/bishop.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';
import 'transformer_encoder2.dart';

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

  static ChessGptFullGame create() {
    final List<List<String>> rawGameSequences = [
      ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5', 'a7a6', 'b5a4', 'g8f6'],
      ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'd2d4', 'c5d4', 'f3d4', 'g8f6'],
    ];

    final (moveStoi, moveItos) = _generateChessVocab(rawGameSequences);
    final (boardStoi, boardItos) = _createBoardVocab();

    const int embedSize = 64;
    const int blockSize = 16;
    const int numLayers = 2;
    const int numHeads = 2;
    const int boardVocabSize = 13; // 12 pieces + 1 empty square
    final int moveVocabSize = moveStoi.length;

    final encoder = TransformerEncoder(
      vocabSize: boardVocabSize,
      embedSize: embedSize,
      blockSize: 64, // Board is always 64 squares
      numLayers: numLayers,
      numHeads: numHeads,
    );

    final decoder = TransformerDecoder(
      vocabSize: moveVocabSize,
      embedSize: embedSize,
      blockSize: blockSize,
      numLayers: numLayers,
      numHeads: numHeads,
      encoderEmbedSize: embedSize,
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

  // CORRECTED: This method now combines the raw game sequences with a recursive search
  static (Map<String, int>, Map<int, String>) _generateChessVocab(
      List<List<String>> gameSequences) {
    final Set<String> allMoves = {};
    for (final sequence in gameSequences) {
      allMoves.addAll(sequence);
    }

    // Recursively generate all possible moves from the initial position
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

    // Generate all moves up to a certain depth.
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

  // Helper map to convert integer piece types to strings
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

    // Padding sequence to block size
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
        await decoder.forward(inputSequence, encoderOutput);
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

    // Truncate if the sequence exceeds the block size
    if (inputSequence.length > blockSize) {
      inputSequence.removeRange(0, inputSequence.length - blockSize);
    }

    // Pad if the sequence is shorter than the block size
    while (inputSequence.length < blockSize) {
      inputSequence.insert(0, padTokenId);
    }

    // Encode the current board state using the game's current FEN
    final boardStateTokens = _tokenizeBoardStateFromFen(game.fen, boardStoi);
    final encoderOutput = encoder.forward(boardStateTokens);

    // Get logits for the next move
    final List<ValueVector> logits =
        await decoder.forward(inputSequence, encoderOutput);
    final ValueVector lastTokenLogits = logits.last;

    // Mask illegal moves
    final List<Move> legalMoves = game.generateLegalMoves();
    final List<int> legalMoveIds =
        legalMoves.map((m) => stoi[game.toAlgebraic(m)]!).toList();

    for (int i = 0; i < lastTokenLogits.values.length; i++) {
      if (!legalMoveIds.contains(i)) {
        lastTokenLogits.values[i].data = -double.infinity;
      }
    }

    // Apply softmax to get probabilities
    final ValueVector probabilities = lastTokenLogits.softmax();

    // Select the best move greedily
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
