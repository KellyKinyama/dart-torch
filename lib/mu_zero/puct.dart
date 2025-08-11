import 'dart:math';

// A conceptual class representing a node in the Monte Carlo Tree.
// In a real implementation, this would contain more complex game-state data.
class Node {
  double value = 0.0;
  final Map<dynamic, double> priorProbabilities = {};
  final Map<dynamic, int> visitCounts = {};
  final Map<dynamic, double> actionValues = {};
  int totalVisits = 0;
}

/// Calculates the PUCT score for a given action from a given node.
/// 
/// - `node`: The current node in the MCTS tree.
/// - `action`: The action being considered.
/// - `c_puct`: The exploration hyperparameter.
double calculatePuctScore(Node node, dynamic action, double c_puct) {
  // Get required values from the node for the given action
  final q_value = node.actionValues[action] ?? 0.0;
  final p_value = node.priorProbabilities[action] ?? 0.0;
  final n_value = node.visitCounts[action] ?? 0;
  final total_n = node.totalVisits;

  // The exploitation term: Q(s,a)
  final exploitationTerm = q_value;

  // The exploration term: c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
  final explorationTerm = c_puct * p_value * sqrt(total_n) / (1 + n_value);

  return exploitationTerm + explorationTerm;
}
