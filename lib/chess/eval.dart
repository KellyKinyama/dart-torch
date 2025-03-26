import 'package:chess/chess.dart';
import 'package:dart_torch/nn/value_vector.dart';

import '../nn/value.dart';

final Map pieceValues = {
  PieceType.PAWN: 1,
  PieceType.KNIGHT: 3,
  PieceType.BISHOP: 3.5,
  PieceType.ROOK: 5,
  PieceType.QUEEN: 9,
  PieceType.KING: 10
};

// simple material based evaluation
double evaluatePosition(Chess c, Color player) {
  if (c.game_over) {
    if (c.in_draw) {
      // draw is a neutral outcome
      return 0.0;
    } else {
      // otherwise must be a mate
      if (c.turn == player) {
        // avoid mates
        return -9999.99;
      } else {
        // go for mating
        return 9999.99;
      }
    }
  } else {
    // otherwise do a simple material evaluation
    var evaluation = 0.0;
    var sq_color = 0;
    for (var i = Chess.SQUARES_A8; i <= Chess.SQUARES_H1; i++) {
      sq_color = (sq_color + 1) % 2;
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }

      final piece = c.board[i];
      if (piece != null) {
        evaluation += (piece.color == player)
            ? pieceValues[piece.type]
            : -pieceValues[piece.type];
      }
    }

    return evaluation;
  }
}

// simple material based evaluation
double evaluatePositionNN(Chess c, Color player) {
  ValueVector vv = ValueVector([]);
  if (c.game_over) {
    if (c.in_draw) {
      // draw is a neutral outcome
      return 0.0;
    } else {
      // otherwise must be a mate
      if (c.turn == player) {
        // avoid mates
        return -9999.99;
      } else {
        // go for mating
        return 9999.99;
      }
    }
  } else {
    // otherwise do a simple material evaluation
    var evaluation = 0.0;
    var sq_color = 0;
    for (var i = Chess.SQUARES_A8; i <= Chess.SQUARES_H1; i++) {
      sq_color = (sq_color + 1) % 2;
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }

      final piece = c.board[i];
      if (piece != null) {
        evaluation += (piece.color == player)
            ? pieceValues[piece.type]
            : -pieceValues[piece.type];

        vv.values.add(pieceValues[piece.type]);
      } else {
        vv.values.add(Value(0.0));
      }
    }

    return evaluation;
  }
}
