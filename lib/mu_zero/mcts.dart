// file: mcts.dart

import '../nn/value_vector.dart';
import '../value.dart';
import 'muzero_model.dart';
import 'package:collection/collection.dart'; // For argmax

class Node {
  final Node? parent;
  final int action; // The action that led to this node
  
  ValueVector hiddenState;
  PredictionOutput prediction; // Policy and Value from the model

  int visitCount = 0;
  double totalValue = 0.0;
  Value reward = Value(0.0);
  Map<int, Node> children = {};

  Node(this.parent, this.action, {required this.hiddenState, required this.prediction});

  double get qValue => visitCount == 0 ? 0.0 : totalValue / visitCount;

  // TODO: Add methods for PUCT score calculation, expansion, etc.
}

/// Runs the MCTS search for a given number of simulations.
Node runMcts(Node rootNode, MuZeroNet model, int numSimulations) {
    for (int i = 0; i < numSimulations; i++) {
        Node node = rootNode;

        // 1. SELECTION: Traverse the tree using PUCT until a leaf node is found
        // TODO: Implement the selection logic.
        // While the node has children, pick the child with the best PUCT score.
        // PUCT formula balances exploration (policy P) and exploitation (value Q).
        
        // 2. EXPANSION: If the node is a leaf, expand it.
        // The policy from the prediction is used to create new child nodes.
        // TODO: Implement expansion. Create children for legal moves.
        
        // 3. BACKPROPAGATION: Update visit counts and values up the tree
        // TODO: Implement backpropagation from the expanded node back to the root.
        // node.visitCount++, node.totalValue += value
    }
    return rootNode;
}

/// Select the best action after the MCTS search is complete.
int selectAction(Node rootNode) {
    // TODO: Based on visit counts or another temperature-based formula,
    // select the most robust move.
    return 0; // Placeholder
}