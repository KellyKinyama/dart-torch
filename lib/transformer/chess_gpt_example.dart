// file: main.dart

import 'dart:io';
import 'package:bishop/bishop.dart';
// import 'chess_gpt_full_game.dart';
import 'chess_gpt2.dart';
import 'transformer_decoder.dart';
import 'transformer_encoder2.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

// Helper function to create the board vocabulary.
// This is a static method from your ChessGptFullGame class, but we need to
// define it here or call it from the static context of the class.
(Map<String, int>, Map<int, String>) _createBoardVocab() {
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

// Function to tokenize a board state from a FEN string.
// This is also a method from your ChessGptFullGame class but is provided here
// for completeness to demonstrate a runnable example.
// List<int> _tokenizeBoardStateFromFen(String fen, Map<String, int> boardStoi) {
//   final game = Game();
//   game.setup(fen: fen);

//   final List<int> tokens = [];
//   final board = game.board;
//   final size = game.size;

//   for (int i = 0; i < size.numIndices; i++) {
//     if (!size.onBoard(i)) continue;

//     Square square = board[i];
//     final piece = square.piece;

//     if (square.isEmpty) {
//       tokens.add(boardStoi['empty']!);
//     } else {
//       final pieceString =
//           piece.type.toString() + (square.colour == Bishop.white ? 'W' : 'B');
//       tokens.add(boardStoi[pieceString]!);
//     }
//   }
//   return tokens;
// }

/// The main function to demonstrate the usage of ChessGptFullGame.
void main() async {
  // 1. Create the model instance using the static factory constructor.
  print('Initializing ChessGptFullGame model...');
  final model = ChessGptFullGame.create();
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
