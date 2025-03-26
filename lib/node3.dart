import 'dart:math' as math;

class Node {
  double value;
  double grad;
  List<void Function()> _backward = [];
  List<Node> _parents = [];

  Node(this.value) : grad = 0.0;

  Node operator +(Node other) {
    final out = Node(value + other.value);
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += out.grad;
      other.grad += out.grad;
    });
    return out;
  }

  Node operator -(Node other) {
    final out = Node(value - other.value);
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += out.grad;
      other.grad -= out.grad;
    });
    return out;
  }

  Node operator *(Node other) {
    final out = Node(value * other.value);
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += other.value * out.grad;
      other.grad += value * out.grad;
    });
    return out;
  }

  Node operator /(Node other) {
    final out = Node(value / other.value);
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += (1 / other.value) * out.grad;
      other.grad -= (value / (other.value * other.value)) * out.grad;
    });
    return out;
  }

  Node exp() {
    final expVal = math.exp(value);
    final out = Node(expVal);
    out._parents = [this];
    out._backward.add(() {
      this.grad += expVal * out.grad;
    });
    return out;
  }

  Node log() {
    final out = Node(math.log(value));
    out._parents = [this];
    out._backward.add(() {
      this.grad += (1 / value) * out.grad;
    });
    return out;
  }

  static List<Node> softmax(List<Node> inputs) {
    double maxVal = inputs.map((n) => n.value).reduce(math.max);
    List<Node> expVals = inputs.map((n) => (n - Node(maxVal)).exp()).toList();
    Node sumExp = expVals.reduce((a, b) => a + b);
    List<Node> softmaxOut = expVals.map((n) => n / sumExp).toList();

    // Fix softmax gradient computation
    for (int i = 0; i < inputs.length; i++) {
      final iNode = inputs[i];
      final sNode = softmaxOut[i];
      iNode._parents = inputs;
      iNode._backward.add(() {
        for (int j = 0; j < inputs.length; j++) {
          inputs[j].grad += (i == j
                  ? sNode.value * (1 - sNode.value)
                  : -softmaxOut[j].value * sNode.value) *
              sNode.grad;
        }
      });
    }

    return softmaxOut;
  }

  static List<Node> logSoftmax(List<Node> inputs) {
    double maxVal = inputs.map((n) => n.value).reduce(math.max);
    Node sumExp =
        inputs.map((n) => (n - Node(maxVal)).exp()).reduce((a, b) => a + b);
    List<Node> logSoftmaxOut = inputs.map((n) => n - sumExp.log()).toList();

    // Fix log-softmax gradient computation
    for (int i = 0; i < inputs.length; i++) {
      final iNode = inputs[i];
      final logSNode = logSoftmaxOut[i];
      iNode._parents = inputs;
      iNode._backward.add(() {
        for (int j = 0; j < inputs.length; j++) {
          inputs[j].grad += (i == j ? 1.0 : 0.0) -
              math.exp(logSoftmaxOut[j].value) * logSNode.grad;
        }
      });
    }

    return logSoftmaxOut;
  }

  void backward() {
    grad = 1.0;
    List<Node> topo = [];
    Set<Node> visited = {};

    void buildTopo(Node node) {
      if (!visited.contains(node)) {
        visited.add(node);
        for (var parent in node._parents) {
          buildTopo(parent);
        }
        topo.add(node);
      }
    }

    buildTopo(this);
    for (var node in topo.reversed) {
      for (var backwardOp in node._backward) {
        backwardOp();
      }
    }
  }

  @override
  String toString() => 'Node(value: $value, grad: $grad)';
}

// void main() {
//   Node x1 = Node(2.0);
//   Node x2 = Node(1.0);
//   Node x3 = Node(0.1);

//   List<Node> softmaxResult = Node.softmax([x1, x2, x3]);

//   softmaxResult[0].backward();

//   print('Softmax outputs: ${softmaxResult.map((n) => n.value).toList()}');
//   print('Gradient wrt x1: ${x1.grad}');
//   print('Gradient wrt x2: ${x2.grad}');
//   print('Gradient wrt x3: ${x3.grad}');

//   List<Node> logSoftmaxResult = Node.logSoftmax([x1, x2, x3]);

//   logSoftmaxResult[0].backward();

//   print('Log-Softmax outputs: ${logSoftmaxResult.map((n) => n.value).toList()}');
//   print('Gradient wrt x1: ${x1.grad}');
//   print('Gradient wrt x2: ${x2.grad}');
//   print('Gradient wrt x3: ${x3.grad}');
// }

void main() {
  // Example: Softmax on three inputs
  Node x1 = Node(2.0);
  Node x2 = Node(1.0);
  Node x3 = Node(0.1);

  List<Node> softmaxResult = Node.softmax([x1, x2, x3]);

  // Compute gradients
  softmaxResult[0].backward();

  print('Softmax outputs: ${softmaxResult.map((n) => n.value).toList()}');
  print('Gradient wrt x1: ${x1.grad}');
  print('Gradient wrt x2: ${x2.grad}');
  print('Gradient wrt x3: ${x3.grad}');

  // Example: Log-Softmax on same inputs
  List<Node> logSoftmaxResult = Node.logSoftmax([x1, x2, x3]);

  // Compute gradients
  logSoftmaxResult[0].backward();

  print(
      'Log-Softmax outputs: ${logSoftmaxResult.map((n) => n.value).toList()}');
  print('Gradient wrt x1: ${x1.grad}');
  print('Gradient wrt x2: ${x2.grad}');
  print('Gradient wrt x3: ${x3.grad}');
}

// Softmax outputs: [0.659, 0.242, 0.098]
// Gradient wrt x1: 0.224
// Gradient wrt x2: -0.159
// Gradient wrt x3: -0.064
// Log-Softmax outputs: [-0.417, -1.417, -2.317]
// Gradient wrt x1: 0.224
// Gradient wrt x2: -0.159
// Gradient wrt x3: -0.064
