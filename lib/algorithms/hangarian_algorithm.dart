import 'dart:collection'; // For Queue
import 'dart:math'; // For min and max

/// Implements the Hungarian Algorithm (Kuhn-Munkres algorithm)
/// to find the minimum cost assignment in a bipartite graph.
///
/// The algorithm finds a perfect matching with the minimum total cost.
/// It works by converting the cost matrix to a profit matrix (multiplying by -1),
/// then finding a maximum weight perfect matching.
class HungarianAlgorithm {
  late List<List<int>> _cost; // The internal cost/profit matrix
  late int _n; // Size of the matrix (number of agents/tasks)

  // Matching information:
  // _xy[x] = y: task 'y' is assigned to agent 'x'
  // _yx[y] = x: agent 'x' is assigned to task 'y'
  late List<int> _xy;
  late List<int> _yx;

  // Labels for vertices:
  // _lx[x]: label for X-vertex (agent) 'x'
  // _ly[y]: label for Y-vertex (task) 'y'
  late List<int> _lx;
  late List<int> _ly;

  // Auxiliary data for the augmenting path algorithm (BFS):
  // _inTreeX[x]: true if agent 'x' is in the current alternating tree
  // _inTreeY[y]: true if task 'y' is in the current alternating tree
  // _prev[x]: parent of agent 'x' in the alternating tree
  // _slack[y]: minimum slack value for task 'y' (not in TreeY) with respect
  //            to agents in TreeX. Slack = lx[x] + ly[y] - cost[x][y]
  // _slackX[y]: the X-vertex 'x' that yields the minimum slack for task 'y'
  late List<bool> _inTreeX;
  late List<bool> _inTreeY;
  late List<int> _prev;
  late List<int> _slack;
  late List<int> _slackX;

  int _matchCount = 0; // Current number of matched pairs

  /// Constructor for the HungarianAlgorithm.
  /// Takes the original cost matrix as input.
  HungarianAlgorithm(List<List<int>> costMatrix) {
    _n = costMatrix.length;

    // Deep copy the cost matrix to avoid modifying the original input
    _cost = List.generate(_n, (i) => List<int>.from(costMatrix[i]));

    // Convert cost matrix to profit matrix by multiplying each element by -1.
    // This allows finding minimum cost by finding maximum profit.
    for (int i = 0; i < _n; i++) {
      for (int j = 0; j < _n; j++) {
        _cost[i][j] *= -1;
      }
    }

    // Initialize all lists with default values
    _xy = List.filled(_n, -1);
    _yx = List.filled(_n, -1);
    _lx = List.filled(_n, 0);
    _ly = List.filled(_n, 0);
    _slack = List.filled(_n, 0);
    _slackX = List.filled(_n, 0);
    _prev = List.filled(_n, 0);
    _inTreeX = List.filled(_n, false);
    _inTreeY = List.filled(_n, false);
  }

  /// Initializes labels for X-vertices (_lx).
  /// For each agent 'i', _lx[i] is set to the maximum cost (profit)
  /// of any task 'j' for that agent.
  void _labelIt() {
    for (int i = 0; i < _n; i++) {
      for (int j = 0; j < _n; j++) {
        _lx[i] = max(_lx[i], _cost[i][j]);
      }
    }
  }

  /// Adds an X-vertex (agent) 'x' to the alternating tree and updates
  /// the slack values for Y-vertices (tasks) not yet in the tree.
  ///
  /// [x]: The agent to add to the tree.
  /// [prevX]: The parent agent of 'x' in the tree.
  void _addTree(int x, int prevX) {
    _inTreeX[x] = true;
    _prev[x] = prevX;
    for (int y = 0; y < _n; y++) {
      // Calculate slack for task 'y' with respect to agent 'x'
      int currentSlack = _lx[x] + _ly[y] - _cost[x][y];
      if (currentSlack < _slack[y]) {
        _slack[y] = currentSlack;
        _slackX[y] = x; // Store 'x' as the agent providing this minimum slack
      }
    }
  }

  /// Updates the labels (_lx and _ly) to create new edges in the equality graph.
  /// This is done when an augmenting path cannot be found with current labels.
  void _updateLabels() {
    // Find the minimum delta (slack) among all Y-vertices not in TreeY
    int delta = 999999999; // A large integer representing infinity

    for (int y = 0; y < _n; y++) {
      if (!_inTreeY[y]) {
        delta = min(delta, _slack[y]);
      }
    }

    // Adjust labels:
    // Subtract delta from labels of X-vertices in TreeX
    for (int x = 0; x < _n; x++) {
      if (_inTreeX[x]) {
        _lx[x] -= delta;
      }
    }

    // Add delta to labels of Y-vertices in TreeY
    for (int y = 0; y < _n; y++) {
      if (_inTreeY[y]) {
        _ly[y] += delta;
      }
    }

    // Subtract delta from slack values of Y-vertices not in TreeY
    for (int y = 0; y < _n; y++) {
      if (!_inTreeY[y]) {
        _slack[y] -= delta;
      }
    }
  }

  /// Finds an augmenting path using a BFS-like approach and updates the matching.
  /// This function is called repeatedly by `findMinCost` until a perfect matching is found.
  void _augment() {
    // If a perfect matching is already found, no need to augment further.
    if (_matchCount == _n) {
      return;
    }

    int x, y = -1, root = -1; // Initialize root with a default value
    Queue<int> q = Queue<int>(); // Queue for BFS traversal

    // Find an exposed (unmatched) X-vertex (agent) to be the root of the alternating tree
    for (int i = 0; i < _n; i++) {
      if (_xy[i] == -1) {
        q.add(root = i);
        _prev[i] = -2; // Sentinel value indicating 'i' is the root
        _inTreeX[i] = true;
        break;
      }
    }

    // Ensure root is assigned before usage
    if (root == -1) {
      throw StateError("No exposed X-vertex found to initialize the root.");
    }

    // Initialize slack values for all Y-vertices with respect to the root
    for (int i = 0; i < _n; i++) {
      _slack[i] = _lx[root] + _ly[i] - _cost[root][i];
      _slackX[i] = root; // 'root' is the X-vertex providing this initial slack
    }

    // Main loop for BFS to find an augmenting path
    while (true) {
      // Build the alternating tree using BFS
      while (q.isNotEmpty) {
        x = q.removeFirst(); // Get current agent from queue

        // Iterate through all tasks 'y'
        for (y = 0; y < _n; y++) {
          // Check if edge (x, y) is in the equality graph (slack is 0)
          // and if 'y' is not already in TreeY
          if ((_lx[x] + _ly[y] - _cost[x][y] == 0) && (!_inTreeY[y])) {
            // If task 'y' is exposed (unmatched), an augmenting path is found!
            if (_yx[y] == -1) {
              break; // Break from inner loop, path found
            }
            // Else, 'y' is matched. Add 'y' to TreeY and its matched agent (_yx[y]) to the queue.
            else {
              _inTreeY[y] = true;
              q.add(_yx[y]); // Add the agent matched with 'y' to the queue
              _addTree(
                  _yx[y], x); // Add edges (x, y) and (y, _yx[y]) to the tree
            }
          }
        }
        if (y < _n) {
          break; // Augmenting path found, break from outer BFS loop
        }
      }

      if (y < _n) {
        break; // Augmenting path found, break from main while(true) loop
      }

      // If no augmenting path found with current labels, update labels
      _updateLabels();

      // After updating labels, check for new edges with slack == 0
      for (y = 0; y < _n; y++) {
        if (!_inTreeY[y] && _slack[y] == 0) {
          // If task 'y' is exposed (unmatched) and now has zero slack,
          // an augmenting path is found!
          if (_yx[y] == -1) {
            x = _slackX[y]; // The X-vertex that caused slack[y] to become 0
            break; // Path found
          }
          // Else, 'y' is matched. Add 'y' to TreeY and its matched agent (_yx[y]) to the queue.
          else {
            _inTreeY[y] = true;
            // Only add to queue if not already in TreeX to avoid cycles
            if (!_inTreeX[_yx[y]]) {
              q.add(_yx[y]);
              _addTree(_yx[y], _slackX[y]);
            }
          }
        }
      }

      if (y < _n) {
        break; // Augmenting path found
      }
    }

    // If an augmenting path was successfully found (y < _n indicates success)
    if (y < _n) {
      _matchCount++; // Increment the count of matched pairs

      // Update the matching along the augmenting path
      // Trace back from 'y' using _prev and _slackX to update _xy and _yx
      for (int cx = _slackX[y], cy = y, ty; cx != -2; cx = _prev[cx], cy = ty) {
        ty = _xy[cx]; // Store the old match of cx
        _xy[cx] = cy; // Assign cy to cx
        _yx[cy] = cx; // Assign cx to cy
      }

      // Reset _inTreeX and _inTreeY for the next augmentation attempt
      _inTreeX = List.filled(_n, false);
      _inTreeY = List.filled(_n, false);

      // Recursively call augment to find next augmenting path until a perfect matching
      // is found or no more paths exist. The `findMinCost` loop also handles this iteratively.
      _augment();
    }
  }

  /// Main method to compute the minimum cost assignment.
  /// Returns the minimum total cost.
  int findMinCost() {
    _labelIt(); // Initialize labels for X-vertices

    // Continue augmenting until a perfect matching is found (matchCount == _n)
    while (_matchCount < _n) {
      // Reset auxiliary structures for each augmentation attempt
      _inTreeX = List.filled(_n, false);
      _inTreeY = List.filled(_n, false);
      _prev = List.filled(_n, 0);
      _slack = List.filled(_n, 999999999); // Reset slack to a large value
      _slackX = List.filled(_n, 0);

      _augment(); // Attempt to find and apply one augmenting path
    }

    int result = 0;
    // Calculate the total profit from the final matching
    for (int i = 0; i < _n; i++) {
      result += _cost[i][_xy[i]];
    }

    // Convert the total profit back to total cost (by negating)
    return -1 * result;
  }
}

/// Example usage of the HungarianAlgorithm.
void main() {
  // First example from the problem description
  List<List<int>> cost1 = [
    [2500, 4000, 3500],
    [4000, 6000, 3500],
    [2000, 4000, 2500]
  ];

  HungarianAlgorithm hungarian1 = HungarianAlgorithm(cost1);
  print(
      "Minimum cost for first example: ${hungarian1.findMinCost()}"); // Expected: 9500

  print('\n---');

  // Second example from the problem description
  List<List<int>> cost2 = [
    [1500, 4000, 4500],
    [2000, 6000, 3500],
    [2000, 4000, 2500]
  ];

  HungarianAlgorithm hungarian2 = HungarianAlgorithm(cost2);
  print(
      "Minimum cost for second example: ${hungarian2.findMinCost()}"); // Expected: 8500
}
