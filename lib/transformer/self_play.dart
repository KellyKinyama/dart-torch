// file: main.dart

import 'dart:math' as math;
import 'package:bishop/bishop.dart';
import 'package:tqdm/tqdm.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';
import 'transformer_encoder.dart';
import 'transformer_decoder_block.dart';
import 'layer_norm2.dart';
import '/nn/layer.dart';
import '/nn/module.dart';

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      if (p.grad != null) {
        p.data -= learningRate * p.grad!;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0;
    }
  }
}

// Helper to create a unified vocabulary
Map<String, int> _generateChessVocab() {
  final Map<String, int> stoi = {};
  int idCounter = 0;

  // Add special tokens first
  stoi['<start>'] = idCounter++;
  stoi['<end>'] = idCounter++;
  stoi['<pad>'] = idCounter++;

  // Generate all legal moves from a starting position to a certain depth
  final board = Game();
  void _generateMovesRecursive(Game currentBoard, int depth) {
    if (depth <= 0) return;
    final moves = currentBoard.generateLegalMoves();
    for (final move in moves) {
      final moveAlg = currentBoard.toAlgebraic(move);
      if (!stoi.containsKey(moveAlg)) {
        stoi[moveAlg] = idCounter++;
      }
      final nextBoard = currentBoard.copy()..makeMove(move);
      _generateMovesRecursive(nextBoard, depth - 1);
    }
  }

  // Generate moves up to a certain depth (e.g., 2)
  _generateMovesRecursive(board, 2);

  // Add some other common moves that might not be in the tree search
  final commonMoves = [
    'e2e4',
    'e7e5',
    'g1f3',
    'b8c6',
    'f1c4',
    'f8c5',
    'c2c3',
    'g8f6',
    'd2d4',
    'e5d4',
    'c3d4',
    'c4b4',
    'd2d4',
    'd7d5',
    'c2c4',
    'e7e6',
    'c1f4',
    'g8f6',
    'g1f3',
    'c7c5',
    'c4d5',
    'f6d5',
    'e3e4',
    'd5f4',
    'd1d8'
  ];
  for (final move in commonMoves) {
    if (!stoi.containsKey(move)) {
      stoi[move] = idCounter++;
    }
  }

  return stoi;
}

Map<String, int> _createBoardVocab() {
  final Map<String, int> stoi = {};
  int idCounter = 0;
  stoi['empty'] = idCounter++;
  for (final color in ['W', 'B']) {
    for (final piece in [
      'pawn',
      'knight',
      'bishop',
      'rook',
      'queen',
      'king'
    ]) {
      stoi['$piece$color'] = idCounter++;
    }
  }
  return stoi;
}

List<int> _tokenizeBoardStateFromFen(String fen, Map<String, int> boardStoi) {
  final game = Game.fromFen(fen);
  final List<int> tokens = [];
  final pieceNames = {
    PieceType.pawn: 'pawn',
    PieceType.knight: 'knight',
    PieceType.bishop: 'bishop',
    PieceType.rook: 'rook',
    PieceType.queen: 'queen',
    PieceType.king: 'king',
  };

  for (int square = 0; square < 64; square++) {
    final piece = game.board.getPiece(square);
    if (piece == null) {
      tokens.add(boardStoi['empty']!);
    } else {
      final colorName = piece.isWhite ? 'W' : 'B';
      final pieceName = pieceNames[piece.type]!;
      final pieceString = '$pieceName$color';
      tokens.add(boardStoi[pieceString]!);
    }
  }
  return tokens;
}

// Wrapper for the full model
class ChessGptFullGame extends Module {
  final TransformerEncoder encoder;
  final TransformerDecoder decoder;
  final Map<String, int> stoi;
  final Map<int, String> itos;
  final Map<String, int> boardStoi;
  final int padTokenId;
  final int startTokenId;
  final int endTokenId;
  final int blockSize;

  ChessGptFullGame(
      this.encoder,
      this.decoder,
      this.stoi,
      this.itos,
      this.boardStoi,
      this.padTokenId,
      this.startTokenId,
      this.endTokenId,
      this.blockSize);

  static ChessGptFullGame create({
    int embedSize = 64,
    int numLayers = 2,
    int numHeads = 2,
    double dropoutRate = 0.1,
  }) {
    final Map<String, int> moveStoi = _generateChessVocab();
    final Map<int, String> moveItos =
        moveStoi.map((key, value) => MapEntry(value, key));
    final Map<String, int> boardStoi = _createBoardVocab();

    const int blockSize = 64;
    final int boardVocabSize = boardStoi.length;
    final int moveVocabSize = moveStoi.length;

    final encoder = TransformerEncoder(
      vocabSize: boardVocabSize,
      embedSize: embedSize,
      blockSize: 64, // Board size is fixed
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
      encoder,
      decoder,
      moveStoi,
      moveItos,
      boardStoi,
      moveStoi['<pad>']!,
      moveStoi['<start>']!,
      moveStoi['<end>']!,
      blockSize,
    );
  }

  @override
  List<Value> parameters() {
    return [...encoder.parameters(), ...decoder.parameters()];
  }
}

// Main function for self-play
void main() async {
  print("--- Generative Pretrained Transformer (GPT) with Self-Play ---");
  final rand = math.Random();

  // Model configuration (can be adjusted)
  const int embedSize = 128;
  const int numLayers = 4;
  const int numHeads = 4;
  const double learningRate = 1e-3;

  // Initialize the model
  final gptModel = ChessGptFullGame.create(
      embedSize: embedSize, numLayers: numLayers, numHeads: numHeads);
  final optimizer = SGD(gptModel.parameters(), learningRate);

  print("\nModel Configuration:");
  print("  Embedding Size: $embedSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");
  print("  Move Vocabulary Size: ${gptModel.stoi.length}");

  // --- Self-Play Training Loop ---
  final int numGames = 1000;
  final int maxMovesPerGame = 100;
  final double explorationTemp = 0.5; // Controls exploration

  print("\n--- Starting Self-Play Training for $numGames games ---");

  final trainingData = [];

  for (int gameNumber in tqdm(List.generate(numGames, (i) => i))) {
    final game = Game();
    List<int> gameHistoryTokens = [gptModel.startTokenId];
    List<String> gameMoves = [];
    final List<String> gameFenHistory = [game.fen];

    while (!game.gameOver && gameMoves.length < maxMovesPerGame) {
      // 1. Get the current board state and history
      final String currentFen = game.fen;

      // 2. Encode the board state
      final List<int> boardStateTokens =
          _tokenizeBoardStateFromFen(currentFen, gptModel.boardStoi);
      final List<ValueVector> encoderOutput =
          gptModel.encoder.forward(boardStateTokens);

      // 3. Prepare decoder input
      List<int> currentInput = List.from(gameHistoryTokens);
      if (currentInput.length > gptModel.blockSize) {
        currentInput =
            currentInput.sublist(currentInput.length - gptModel.blockSize);
      }

      // 4. Forward pass through the decoder
      final List<ValueVector> logits =
          gptModel.decoder.forward(currentInput, encoderOutput);
      final ValueVector lastTokenLogits = logits.last;

      // 5. Get legal moves and apply masking
      final List<Move> legalMoves = game.generateLegalMoves();
      final Set<int> legalMoveTokenIds = legalMoves
          .map((m) => gptModel.stoi[game.toAlgebraic(m)])
          .whereType<int>()
          .toSet();

      final List<Value> maskedLogits = List.generate(gptModel.stoi.length, (index) {
        if (legalMoveTokenIds.contains(index)) {
          return lastTokenLogits.values[index];
        }
        return Value(double.negativeInfinity);
      });

      // 6. Apply temperature for exploration and sample a move
      final ValueVector probabilities =
          ValueVector(maskedLogits).softmax(temperature: explorationTemp);

      final List<double> probList =
          probabilities.values.map((v) => v.data).toList();
      int predictedMoveToken = _sampleFromDistribution(probList, rand);

      // 7. Make the move and update state
      final String? nextMoveAlgebraic = gptModel.itos[predictedMoveToken];
      final Move? move =
          nextMoveAlgebraic != null ? game.getMove(nextMoveAlgebraic) : null;

      if (move != null) {
        game.makeMove(move);
        gameHistoryTokens.add(predictedMoveToken);
        gameMoves.add(nextMoveAlgebraic!);
        gameFenHistory.add(game.fen);
      } else {
        // If an illegal move is sampled, break and consider the game a draw
        break;
      }
    }

    // 8. Determine outcome and add to training data
    final double outcome = _getOutcome(game.result);
    if (gameMoves.isNotEmpty) {
      trainingData.add({
        'fens': gameFenHistory,
        'moves': gameMoves,
        'outcome': outcome,
      });
    }

    // 9. Train on the collected game data
    if (trainingData.isNotEmpty) {
      train(trainingData.first, optimizer, gptModel, gptModel.stoi);
      trainingData.clear(); // Clear data after training on it
    }
  }
}

// Helper function to sample from a probability distribution
int _sampleFromDistribution(List<double> probabilities, math.Random rand) {
  double sum = 0.0;
  for (final p in probabilities) {
    sum += p;
  }
  double r = rand.nextDouble() * sum;
  for (int i = 0; i < probabilities.length; i++) {
    r -= probabilities[i];
    if (r <= 0) return i;
  }
  return probabilities.length - 1;
}

// Helper to get a numeric outcome from a game result
double _getOutcome(Result? result) {
  if (result == null) return 0.0;
  if (result.winner == Color.white) return 1.0;
  if (result.winner == Color.black) return -1.0;
  return 0.0;
}

// The core training function
void train(
    Map<String, dynamic> gameData,
    SGD optimizer,
    ChessGptFullGame gptModel,
    Map<String, int> stoi) {
  final List<String> fens = gameData['fens'] as List<String>;
  final List<String> moves = gameData['moves'] as List<String>;
  final double outcome = gameData['outcome'] as double;
  final int padTokenId = stoi['<pad>']!;
  final int startTokenId = stoi['<start>']!;

  // We train on the entire game sequence
  final List<int> tokenizedSeq = [
    startTokenId,
    ...moves.map((m) => stoi[m]!).where((id) => id != null)
  ];

  optimizer.zeroGrad();
  Value totalLoss = Value(0.0);

  for (int i = 0; i < fens.length - 1; i++) {
    final inputMoveSequence = tokenizedSeq.sublist(0, i + 1);
    final targetMoveToken = tokenizedSeq[i + 1];

    final String fen = fens[i];
    final List<int> boardStateTokens =
        _tokenizeBoardStateFromFen(fen, gptModel.boardStoi);

    // Pad the input move sequence
    final List<int> paddedInput = List.filled(gptModel.blockSize, padTokenId)
      ..setRange(
          gptModel.blockSize - inputMoveSequence.length,
          gptModel.blockSize,
          inputMoveSequence.sublist(
              math.max(0, inputMoveSequence.length - gptModel.blockSize)));

    final List<ValueVector> encoderOutput =
        gptModel.encoder.forward(boardStateTokens);
    final List<ValueVector> logits =
        gptModel.decoder.forward(paddedInput, encoderOutput);

    // Policy Loss (Cross-Entropy)
    final ValueVector tokenLogits = logits.last;
    final Value trueLogit = tokenLogits.values[targetMoveToken];
    final Value sumExpLogits =
        tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
    final Value logSumExp = sumExpLogits.log();
    final Value policyLoss = logSumExp - trueLogit;

    // Value Loss (MSE)
    final Value predictedValue = gptModel.decoder
        .forward(paddedInput, encoderOutput)
        .last; // Needs to be re-run to get the Value head output. A more efficient way is to modify the decoder to return both.
    final Value valueLoss = (predictedValue - Value(outcome)).pow(2);

    totalLoss += policyLoss + valueLoss;
  }
  
  if (fens.length > 1) {
    totalLoss = totalLoss / Value((fens.length - 1).toDouble());
    totalLoss.backward();
    optimizer.step();
  }
}