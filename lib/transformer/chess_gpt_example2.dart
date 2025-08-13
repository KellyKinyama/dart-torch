// file: main.dart

import 'package:bishop/bishop.dart';
import 'chess_gpt4.dart';

void main() async {
  print('Initializing ChessGptFullGame model...');
  final model = ChessGptFullGame.create();
  print('Model initialization complete.\n');

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
  final double whiteWins = 1.0;

  print('Starting training on a sample game...');
  await model.train(sampleGame, whiteWins);
  print('Training complete!\n');

  final Game gameToPredict = Game();
  gameToPredict.makeMove(gameToPredict.getMove('e2e4')!);

  print('Predicting the next move from FEN: ${gameToPredict.fen}');
  final (predictedMove, predictedValue) =
      await model.predictNextMove(gameToPredict);
  print('Predicted best move: $predictedMove');
  print('Predicted position value: $predictedValue');
}
