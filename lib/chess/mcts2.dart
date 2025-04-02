import 'dart:async';
import 'dart:math' as Math;
import 'package:chess/chess.dart';

import 'nn.dart';
import '../nn/value.dart';
import '../nn/value_vector.dart';
import 'eval.dart';

import 'dart:math';

const VALUE_KNOWN_WIN = 9999.99;
double valueToReward(int value) {
  // Equivalent of `const double k = -0.00490739829861;`
  const double k = -0.00490739829861;
  double r = 1.0 / (1.0 + exp(k * value));

  // Optionally clamp reward (if you have REWARD_MATED or REWARD_MATE defined)
  r = r.clamp(0.0, 1.0);
  return r;
}

int rewardToValue(double reward) {
  // Early clamping for extreme reward values
  if (reward > 0.99) return VALUE_KNOWN_WIN.toInt();
  if (reward < 0.01) return -VALUE_KNOWN_WIN.toInt();

  const double g = 203.77396313709564; // inverse of k
  double value = g * log(reward / (1.0 - reward));
  return value.round();
}

class Edge {
  Move move;
  double? score;
  int visits = 0;
  Edge(this.move, {this.score});

  int depth = 0;
}

class Node {
  String key;
  List<Edge>? edges;
  Node? parent;
  double? score;

  int nodeVisits = 0;
  Node(this.key, {this.edges});

  Edge? currentEdge;
}

Map<String, List<Node>> mctsTree = {};

Node getNode(String key) {
  // print("Getting node: $key");
  final hashkey = key.substring(0, 20);

  if (mctsTree[hashkey] == null) {
    mctsTree[hashkey] = [];
    final node = Node(key);
    mctsTree[hashkey]!.add(node);
    return node;
  } else {
    final nodes = mctsTree[hashkey];
    for (Node mctsNode in nodes!) {
      if (mctsNode.key == key) {
        return mctsNode;
      }
    }
    final node = Node(key);
    mctsTree[hashkey]!.add(node);

    return node;
  }
}

class PolicyMapper {
  int _cols;
  int _rows;
  late ValueVector data;

  PolicyMapper(this._rows, this._cols);

  Value at(int row, int col) {
    return data.values[row * _cols + col];
  }

  // Convert the 1D index back to a (from, to) pair
  (int from, int to) fromIndex(int index) {
    int from = index ~/ _cols; // Integer division to get the row (from square)
    int to = index % _cols; // Modulo to get the column (to square)
    return (from, to);
  }
}

class Mcts {
  late Node _currentNode;
  late Node _rootNode;

  final lr = 0.02;

  late final MultiLayerPerceptron model; // 4 inputs → 2 outputs
  Mcts(this._chess) {
    final initVector = initEval(_chess, _chess.turn);
    vv = initVector;
    model = MultiLayerPerceptron(lr); // 4 inputs → 2 outputs

    print("init vector: ${vv.values.length}");
  }
  Chess _chess;
  int ply = 0;

  ValueVector vv = ValueVector([]);
  Move monteCarlo(Chess chess) {
    _chess = chess;
    createRoot(_chess);

    while (currentNode().nodeVisits < 50) {
      treePolicy();
      currentNode().score = playOutPolicy();
      backup(currentNode().score!);
      // emitPV();
    }
    print("Current node: ${_rootNode.key}");
    print("FEN:          ${_chess.generate_fen()}");
    print("Moves: ${_chess.san_moves()}");

    return bestMove().move;
  }

  void createRoot(Chess chess) {
    _rootNode = _currentNode = getNode(chess.generate_fen());
    _currentNode.edges = [];
    generateMoves();
    _currentNode.nodeVisits++;
    ply = 0;
  }

  bool isRoot() {
    return _rootNode.key == _currentNode.key;
  }

  void treePolicy() {
    for (Edge edge in currentNode().edges!) {
      Chess c = _chess.copy();

      bool completedIteration = false;

      int? currentScore;

      final duration = Duration(milliseconds: 10);

      Timer.periodic(duration, (timer) {
        c.move(edge.move);
        int depth = edge.depth++;

        final score = pvSearch(c, depth, 9999999.0, -9999999.0, c.turn);
        c.undo();
      });
    }
    // while (currentNode().nodeVisits != 0) {
    //   Edge m = bestMove(uct: true);

    //   currentNode().currentEdge = m;

    //   currentNode().nodeVisits++;

    //   print("Continuing with alphabeta ${currentNode().nodeVisits}");
    //   doMove(m.move);
    // }
  }

  void emitPV() {
    Node prevNode = currentNode();
    while (!isRoot()) {
      prevNode = currentNode();
      undoMove();
    }
  }

  // void rootMoves(){
  //   for(int id=1;id<120;id)
  // }

  void backup(double reward,
      {bool abMode = false, double backupMinimax = 0.0}) {
    assert(ply >=
        1); // You must maintain a ply counter externally (like stack depth)

    double weight = 1.0;

    while (!isRoot()) {
      undoMove(); // Go up one level in tree
      reward = 1.0 - reward; // Flip perspective for opponent

      Node node = currentNode();
      Edge? edge = node.currentEdge;
      if (edge == null) continue;

      if (abMode) {
        edge.score = reward;
        abMode = false;
      }

      // Compensate virtual loss (if used before in selection phase)
      edge.visits = edge.visits - 1;

      // Update statistics
      edge.visits += weight.toInt();

      // If using separate `actionValue`, `meanActionValue`, you can add those fields
      if (edge.score == null) edge.score = 0.0;
      edge.score = edge.score! + weight * reward; // This becomes action value

      double meanActionValue = edge.score! / edge.visits;

      // Ensure valid range
      assert(meanActionValue >= 0.0 && meanActionValue <= 1.0);

      // Optional minimax backup
      if (backupMinimax > 0.0 && node.edges != null && node.edges!.isNotEmpty) {
        final bestEdge = node.edges!.reduce((a, b) =>
            (a.score ?? 0.0) / (a.visits == 0 ? 1 : a.visits) >
                    (b.score ?? 0.0) / (b.visits == 0 ? 1 : b.visits)
                ? a
                : b);

        double bestMean = (bestEdge.score ?? 0.0) /
            (bestEdge.visits == 0 ? 1 : bestEdge.visits);
        reward = reward * (1.0 - backupMinimax) + bestMean * backupMinimax;
      }
    }

    // assert(isRoot());
  }

  // Edge bestMove({bool uct = false}) {
  //   final moveEvalPairs = currentNode().edges;
  //   Edge bestEdge = moveEvalPairs![0];

  //   if (uct) {
  //     for (final l in moveEvalPairs) {
  //       if (l.visits < bestEdge.visits) {
  //         bestEdge = l;
  //       }
  //     }
  //   } else {
  //     for (final l in moveEvalPairs) {
  //       if (l.score! > bestEdge.score!) {
  //         bestEdge = l;
  //       }
  //     }
  //   }
  //   return bestEdge;
  // }

  Edge bestMove({bool uct = false, double explorationConstant = 1.41}) {
    final moveEvalPairs = currentNode().edges;
    Edge bestEdge = moveEvalPairs![0];

    if (uct) {
      final parentVisits = currentNode().nodeVisits;
      double bestUCT = double.negativeInfinity;

      for (final edge in moveEvalPairs) {
        final edgeVisits = edge.visits;
        final averageScore = edge.score ?? 0.0;
        double uctValue;

        if (edgeVisits == 0) {
          // Encourage exploration of unvisited nodes
          uctValue = double.infinity;
        } else {
          uctValue = averageScore +
              explorationConstant *
                  (Math.sqrt(Math.log(parentVisits + 1) / edgeVisits));
        }

        if (uctValue > bestUCT) {
          bestUCT = uctValue;
          bestEdge = edge;
        }
      }
    } else {
      for (final edge in moveEvalPairs) {
        if ((edge.score ?? 0.0) > (bestEdge.score ?? 0.0)) {
          bestEdge = edge;
        }
      }
    }

    return bestEdge;
  }

  Node currentNode() {
    return _currentNode;
  }

  void doMove(Move move) {
    _chess.move(move);
    _currentNode = getNode(_chess.generate_fen());
  }

  void undoMove() {
    _chess.undo();
    _currentNode = getNode(_chess.generate_fen());
  }

  double playOutPolicy() {
    generateMoves();
    if (currentNode().edges!.isEmpty) {
      return -9999.99;
    }
    return bestMove().score!;
  }

  void backpropagate(List<Edge> path, double finalReward,
      {double alpha = 0.1, double gamma = 0.95}) {
    double reward = finalReward;

    for (int i = path.length - 1; i >= 0; i--) {
      final edge = path[i];
      edge.visits += 1;

      // Default score if null
      edge.score ??= 0.0;

      // Estimate max Q-value of next state (if available)
      double nextMaxQ = 0.0;
      if (i < path.length - 1) {
        final nextEdge = path[i + 1];
        nextMaxQ = nextEdge.score ?? 0.0;
      }

      // Q-learning update rule
      edge.score =
          edge.score! + alpha * (reward + gamma * nextMaxQ - edge.score!);

      // Update reward to backpropagate
      reward = edge.score!;
    }
  }

  void generateMoves() {
    final toPlay = _chess.turn;
    currentNode().edges = [];

    final encoded = eval(_chess, toPlay);
    model.zeroGrad();
    final data = model.forward(encoded);
    data.then((onValue) {
      print("Encoded ouptput: ${onValue}:");
    });
    for (final m in _chess.moves({'asObjects': true}) as Iterable<Move>) {
      _chess.move(m);
      double prior = valueToReward(evaluatePosition(_chess, toPlay).toInt());
      currentNode().edges!.add(Edge(m, score: prior));
      _chess.undo();
    }
    if (currentNode().edges!.isEmpty) {
      print("No moves found");
    }
    currentNode().nodeVisits++;
  }

  // simple material based evaluation
  ValueVector initEval(Chess c, Color player) {
    ValueVector vv = ValueVector([]);
    if (c.game_over) {
      if (c.in_draw) {
        // draw is a neutral outcome
        throw "error";
      } else {
        // otherwise must be a mate
        if (c.turn == player) {
          // avoid mates
          throw "error";
        } else {
          // go for mating
          throw "error";
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

          vv.values.add(Value(pieceValues[piece.type].toDouble()));
        } else {
          vv.values.add(Value(0.0));
        }
      }

      player == Chess.WHITE
          ? vv.values.add(Value(0.0))
          : vv.values.add(Value(1.0));
    }
    return vv;
  }

  ValueVector encoded() {
    ValueVector encoded = ValueVector(
        List.generate(65, (int index) => Value(vv.values[index].data)));
    return encoded;
  }

  ValueVector eval(Chess c, Color player) {
    ValueVector encodedBoad = encoded();
    if (c.game_over) {
      if (c.in_draw) {
        // draw is a neutral outcome
        throw "error";
      } else {
        // otherwise must be a mate
        if (c.turn == player) {
          // avoid mates
          throw "error";
        } else {
          // go for mating
          throw "error";
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

          encodedBoad.values.add(Value(pieceValues[piece.type].toDouble()));
        } else {
          encodedBoad.values.add(Value(0.0));
        }
      }

      player == Chess.WHITE
          ? vv.values.add(Value(0.0))
          : vv.values.add(Value(1.0));
    }
    return encodedBoad;
  }
}

// implements a simple alpha beta algorithm
double alphaBeta(Chess c, int depth, double alpha, double beta, Color player) {
  if (depth == 0 || c.game_over) {
    return evaluatePosition(c, player);
  }

  // go through all legal moves
  for (final m in c.moves({'asObjects': true}) as Iterable<Move>) {
    c.move(m);
    final val = -alphaBeta(c, depth - 1, -beta, -alpha, c.turn);
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

// implements a simple alpha beta algorithm
Stream<double> pvSearch(
    Chess c, int depth, double alpha, double beta, Color player) async* {
  if (depth == 0 || c.game_over) {
    yield evaluatePosition(c, c.turn);
  }

  // if the computer is the current player

  // go through all legal moves
  for (final m in c.moves({'asObjects': true}) as Iterable<Move>) {
    c.move(m);
    await for (double val in pvSearch(c, depth - 1, -beta, -alpha, c.turn)) {
      val = -val;
      if (val >= beta) {
        yield beta;
      }
      if (val > alpha) {
        alpha = val;
      }
    }
    c.undo();

  }
  yield alpha;
}
