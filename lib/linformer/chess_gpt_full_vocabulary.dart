// file: main.dart

import 'package:bishop/bishop.dart';
import 'chess_gpt3.dart';

// Helper function to create the board vocabulary for the encoder.
(Map<String, int>, Map<int, String>) createBoardVocab() {
  final Map<String, int> stoi = {};
  int idCounter = 0;
  stoi['empty'] = idCounter++;
  for (var color in ['W', 'B']) {
    for (var piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']) {
      stoi[piece + color] = idCounter++;
    }
  }
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));
  return (stoi, itos);
}

// Helper function to create the move vocabulary for the decoder.
(Map<String, int>, Map<int, String>) createMoveVocab() {
  // Removed '<start>'
  final Map<String, int> stoi = {'<pad>': 0, '<end>': 1};
  int idCounter = 2; // Adjusted counter

  final List<String> files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  final List<String> ranks = ['1', '2', '3', '4', '5', '6', '7', '8'];
  final Map<int, String> indexToSquare = {};

  int index = 0;
  for (int r = 0; r < ranks.length; r++) {
    for (int f = 0; f < files.length; f++) {
      String square = files[f] + ranks[r];
      indexToSquare[index] = square;
      index++;
    }
  }

  for (int fromIdx = 0; fromIdx < 64; fromIdx++) {
    for (int toIdx = 0; toIdx < 64; toIdx++) {
      if (fromIdx == toIdx) {
        continue;
      }

      String fromSquare = indexToSquare[fromIdx]!;
      String toSquare = indexToSquare[toIdx]!;
      String moveToken = fromSquare + toSquare;
      stoi[moveToken] = idCounter++;
    }
  }

  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));
  return (stoi, itos);
}

/// The main function to demonstrate the usage of ChessGptFullGame.
void main() async {
  // 1. Create the model instance using the static factory constructor.
  print('Initializing ChessGptFullGame model...');
  final model = ChessGptFullGame.create(
    projK: 128,
    moveVocab: createMoveVocab(), // Pass the move vocabulary
    boardVocab: createBoardVocab(), // Pass the board vocabulary
  );
  print('Model initialization complete.\n');

  // 2. Define a sample training game.
  final List<String> sampleGame = [
    'e2e4',
    'e7e5',
    'g1f3',
    'b8c6',
    'f1b5',
    'a7a6',
    'b5a4',
    'g8f6'
  ];

  // 3. Train the model on the sample game.
  print('Starting training on a sample game...');
  await model.train(sampleGame);
  print('Training complete!\n');

  // 4. Set up a new game position for prediction.
  final Game gameToPredict = Game();
  // Play the first move to create a new board state for the model to analyze.
  gameToPredict.makeMove(gameToPredict.getMove('e2e4')!);

  // 5. Predict the next move from the current game position.
  print('Predicting the next move from FEN: ${gameToPredict.fen}');
  final String predictedMove = await model.predictNextMove(gameToPredict);
  print('Predicted best move: $predictedMove');
}
