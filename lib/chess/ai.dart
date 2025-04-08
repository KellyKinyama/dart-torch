// Example of a simple chess AI in Dart using chess.dart
// The user plays white by default.
import 'dart:async';
import 'dart:io';
import 'dart:math';
import "package:chess/chess.dart";
// import 'chess_board.dart';

// Find the best move using async pvSearch (parallel execution)
// Future<void> findBestMove(Chess chess, void Function(Move?) onMoveFound) async {
//   const PLY = 4;
//   final alpha = -9999999.0;
//   final beta = 9999999.0;
//   final totalMoves = chess.moves({'asObjects': true}) as Iterable<Move>;

//   print("Starting parallel pvSearch for all moves...");
//   final startTime = DateTime.now();

//   // Launch all searches in parallel
//   final futures = <Future<List>>[];

//   for (final m in totalMoves) {
//     chess.move(m);
//     final c = Chess.fromFEN(chess.fen);
//     chess.undo();

//     // Log when the search starts
//     print("[${DateTime.now()}] Starting search for move: $m");

//     // Store the move and its evaluation as a Future
//     futures.add(
//       pvSearch(c, PLY, -beta, -alpha, c.turn).then((eval) {
//         // print(
//         //     "[${DateTime.now()}] Finished search for move: $m with eval: $eval");
//         return [m, -eval];
//       }),
//     );
//   }

//   // Wait for all searches to complete
//   final moveEvalPairs = await Future.wait(futures);
//   final endTime = DateTime.now();

//   // Calculate total time
//   print(
//       "All searches completed in ${endTime.difference(startTime).inMilliseconds} ms");

//   // Determine the best move
//   Move? bestMove;
//   var highestEval = -9999999.0;
//   for (final l in moveEvalPairs) {
//     if (l[1] > highestEval) {
//       highestEval = l[1];
//       bestMove = l[0] as Move;
//     }
//   }

//   print("Picking best move: $bestMove");
//   onMoveFound(bestMove); // Callback with the best move
// }

// Future<void> findBestMove(Chess chess, void Function(Move?) onMoveFound) async {
//   const PLY = 4;
//   final alpha = -9999999.0;
//   final beta = 9999999.0;
//   Move? bestMove;

//   print("Starting parallel pvSearch with iterative deepening...");

//   for (int depth = 1; depth <= PLY; depth++) {
//     final startTime = DateTime.now();
//     final totalMoves = chess.moves({'asObjects': true}) as List<Move>;

//     // Sort moves based on previous best move heuristic
//     if (bestMove != null) {
//       totalMoves.sort((a, b) => (a == bestMove)
//           ? -1
//           : (b == bestMove)
//               ? 1
//               : 0);
//     }

//     final futures = <Future<List>>[];

//     for (final m in totalMoves) {
//       chess.move(m);
//       final c = Chess.fromFEN(chess.fen);
//       // chess.undo();

//       print("[${DateTime.now()}] Starting search (depth=$depth) for move: $m");

//       futures.add(
//         pvSearch(c, depth, -beta, -alpha, c.turn).then((eval) => [m, -eval]),
//       );
//       chess.undo();
//     }

//     // Wait for all searches to complete
//     final moveEvalPairs = await Future.wait(futures);
//     final endTime = DateTime.now();

//     print(
//         "Depth $depth completed in ${endTime.difference(startTime).inMilliseconds} ms");

//     // Find the best move at this depth
//     Move? currentBestMove;
//     var highestEval = -9999999.0;
//     for (final l in moveEvalPairs) {
//       if (l[1] > highestEval) {
//         highestEval = l[1];
//         currentBestMove = l[0] as Move;
//       }
//     }

//     if (currentBestMove != null) {
//       bestMove = currentBestMove;
//     }

//     print("Best move at depth $depth: $bestMove");
//   }

//   print("Final best move: $bestMove");
//   onMoveFound(bestMove);
// }

// Future<void> findBestMove(Chess chess, void Function(Move?) onMoveFound) async {
//   const PLY = 5;
//   final alpha = -9999999.0;
//   final beta = 9999999.0;
//   Move? bestMove;
//   var highestEval = -9999999.0;

//   print("Starting parallel pvSearch with iterative deepening...");

//   for (int depth = 1; depth <= PLY; depth++) {
//     final startTime = DateTime.now();
//     final totalMoves = chess.moves({'asObjects': true}) as List<Move>;

//     // Sort moves based on previous best move heuristic
//     if (bestMove != null) {
//       totalMoves.sort((a, b) => (a == bestMove)
//           ? -1
//           : (b == bestMove)
//               ? 1
//               : 0);
//     }

//     print("Starting searches at depth $depth...");

//     for (final m in totalMoves) {
//       chess.move(m);
//       final c = Chess.fromFEN(chess.fen);
//       chess.undo();

//       print("[${DateTime.now()}] Starting search (depth=$depth) for move: $m");

//       pvSearch(c, depth, -beta, -alpha, c.turn).then((eval) {
//         final evalScore = -eval; // Negamax result
//         print(
//             "[${DateTime.now()}] Finished search (depth=$depth) for move: $m with eval: $evalScore");

//         // Check if this move is better
//         if (evalScore > highestEval) {
//           highestEval = evalScore;
//           bestMove = m;
//           print(
//               "Updated best move at depth $depth: $bestMove with eval: $highestEval");
//         }
//       });
//     }

//     final endTime = DateTime.now();
//     print(
//         "Depth $depth initiated in ${endTime.difference(startTime).inMilliseconds} ms");
//   }

//   // Give some time for searches to finish before calling the final best move.
//   await Future.delayed(Duration(milliseconds: 60000)); // Adjust delay if needed

//   print("Final best move: $bestMove");
//   onMoveFound(bestMove);
// }

Future<void> findBestMove(Chess chess, void Function(Move?) onMoveFound) async {
  const PLY = 6;
  final alpha = -9999999.0;
  final beta = 9999999.0;
  Move? bestMove;
  var highestEval = -9999999.0;

  print("Starting parallel pvSearch with iterative deepening...");

  Future<void> searchMove(Move m, int depth) async {
    //chess.move(m);
    // final c = Chess.fromFEN(chess.fen);
    final c = chess.copy();
    //chess.undo();

    print("[${DateTime.now()}] Starting search (depth=$depth) for move: $m");

    c.move(m);
    final eval = -await pvSearch(c, depth, -beta, -alpha);
    c.undo();
    print(
        "[${DateTime.now()}] Finished search (depth=$depth) for move: $m with eval: $eval");

    // Check if this move is the new best move
    if (eval > highestEval) {
      highestEval = eval;
      bestMove = m;
      print(
          "Updated best move at depth $depth: $bestMove with eval: $highestEval");
    }
    if (depth > PLY) return; // Stop at max depth
    // Immediately start searching this move at the next depth
    if (m == null) throw Exception("move cannot be null");
    await searchMove(m, depth + 1);
  }

  final totalMoves = chess.moves({'asObjects': true}) as List<Move>;

  for (final m in totalMoves) {
    searchMove(m, 1); // Start iterative deepening for each move
  }

  // Give some time for searches to finish before calling the final best move.
  await Future.delayed(
      Duration(seconds: 10)); // Adjust based on search depth/time

  print("Final best move: $bestMove");
  onMoveFound(bestMove);
}

// Ensure pvSearch is truly async
Future<double> pvSearch(Chess c, int depth, double alpha, double beta) async {
  if (depth == 0 || c.game_over) {
    return evaluatePosition(c, c.turn);
  }

  for (final m in c.moves({'asObjects': true}) as Iterable<Move>) {
    c.move(m);

    // Ensure async execution
    // await Future(() {});

    final val = -await alphaBeta(c, depth - 1, -beta, -alpha, c.turn);
    c.undo();

    if (val >= beta) {
      return beta;
    }
    if (val > alpha) {
      // print("found best move: $m with eval: $val");
      alpha = val;
    }
  }
  return alpha;
}

// Make alphaBeta async to avoid blocking execution
Future<double> alphaBeta(
    Chess c, int depth, double alpha, double beta, Color player) async {
  if (depth == 0 || c.game_over) {
    // return evaluatePosition(c, c.turn);
    return await qSearch(c, depth, alpha, beta, c.turn);
  }

  // final them = c.turn;

  // bool inCheck = c.attacked(
  //     them == Chess.WHITE ? Chess.BLACK : Chess.WHITE, c.kings[c.turn]);
  // if (inCheck) print("We are in check");

  for (final m in c.moves({'asObjects': true}) as Iterable<Move>) {
    c.move(m);

    // if (m.captured == null) {
    //   if (c.attacked(them, c.kings[c.turn])) {
    //     print("This move gives check");
    //   }
    // }
    // if (m.captured != null) {
    //   print("This move is a capture");
    // }
    // Ensure async execution
    // await Future(() {});

    final val = -await alphaBeta(c, depth - 1, -beta, -alpha, c.turn);
    c.undo();

    if (val >= beta) {
      return beta;
    }
    if (val > alpha) {
      alpha = val;
    }
  }
  return alpha;
}

// Make alphaBeta async to avoid blocking execution
Future<double> qSearch(
    Chess c, int depth, double alpha, double beta, Color player) async {
  // if (c.game_over) {

  bool inCheck = c.in_check;

  // final standPat = qPosition(c, c.turn);
  final standPat = evaluatePosition(c, c.turn);

  if (c.game_over) {
    return standPat;
  } else {
    // return standPat;
    if (standPat >= beta) return standPat;
    if (standPat <= alpha) return standPat;
  }

  int movesFound = 0;

  for (final m in c.moves({'asObjects': true}) as Iterable<Move>) {
    bool problematicMove = false;
    c.move(m);

    // if (inCheck) {
    //   problematicMove = true;
    // } //else
    if (m.captured != null) {
      problematicMove = true;
    }
    // else
    // if (c.in_check) {
    //   problematicMove = true;
    // }
    if (!problematicMove) {
      c.undo();
      continue;
    }
    // // Ensure async execution
    // await Future(() {});

    final val = -await qSearch(c, depth - 1, -beta, -alpha, c.turn);
    movesFound++;
    c.undo();

    if (val >= beta) {
      return beta;
    }
    if (val > alpha) {
      alpha = val;
    }
  }

  if (movesFound == 0) return standPat;
  return alpha;
}

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
double qPosition(Chess c, Color player) {
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

Future<void> main() async {
  final chess = Chess();
  // Mcts mcts = Mcts(chess);
  while (!chess.game_over) {
    print(chess.ascii);
    print('What would you like to play?');
    final playerMove = stdin.readLineSync();
    if (chess.move(playerMove) == false) {
      print(
          "Could not understand your move or it's illegal. Moves should be in SAN format.");
      continue;
    }
    print('Computer thinking...');

    // final compMove = findBestMove(chess);
    // final compMove = mcts.monteCarlo(chess);

    await findBestMove(chess, (bestMove) {
      print("Move: $bestMove");

      // compMove.then((onMove) {
      if (bestMove != null) {
        print("Move: $bestMove");
        print('Computer Played: ${chess.move_to_san(bestMove)}');
        chess.move(bestMove);
      }

      print(chess.ascii);
      if (chess.in_checkmate) {
        print('Checkmate');
      }
      if (chess.in_stalemate) {
        print('Stalemate');
      }
      if (chess.in_draw) {
        print('Draw');
      }
      if (chess.insufficient_material) {
        print('Insufficient Material');
      }
    });

    // });
  }
}
