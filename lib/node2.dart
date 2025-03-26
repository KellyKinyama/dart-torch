import 'dart:math' as math;

// final tanhVal = (math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value));

class Node {
  double value;
  double grad;
  List<void Function()> _backward = [];
  List<Node> _parents = [];

  Node(this.value) : grad = 0.0;

  Node operator *(Node other) {
    final out = Node(value * other.value);
    out._parents = [this, other];
    out._backward.add(() {
      grad += other.value * out.grad;
      other.grad += value * out.grad;
    });
    return out;
  }

  Node operator +(Node other) {
    final out = Node(value + other.value);
    out._parents = [this, other];
    out._backward.add(() {
      grad += out.grad;
      other.grad += out.grad;
    });
    return out;
  }

  Node exp() {
    final expVal = math.exp(value);
    final out = Node(expVal);
    out._parents = [this];
    out._backward.add(() {
      grad += expVal * out.grad;
    });
    return out;
  }

  Node relu() {
    final out = Node(value > 0 ? value : 0);
    out._parents = [this];
    out._backward.add(() {
      grad += (value > 0 ? 1.0 : 0.0) * out.grad;
    });
    return out;
  }

  Node sigmoid() {
    final sigVal = 1.0 / (1.0 + math.exp(-value));
    final out = Node(sigVal);
    out._parents = [this];
    out._backward.add(() {
      grad += sigVal * (1 - sigVal) * out.grad;
    });
    return out;
  }

  Node tanh() {
    final expPos = math.exp(value);
    final expNeg = math.exp(-value);
    final tanhVal = (expPos - expNeg) / (expPos + expNeg);
    final out = Node(tanhVal);
    out._parents = [this];
    out._backward.add(() {
      grad += (1 - tanhVal * tanhVal) * out.grad;
    });
    return out;
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

void main() {
  // Example: Compute derivative of f(x) = ReLU(x) at x = -1 and x = 2
  Node x1 = Node(-1.0);
  Node x2 = Node(2.0);
  Node f1 = x1.relu(); // f(x) = max(0, x)
  Node f2 = x2.relu();

  f1.backward();
  f2.backward();

  print(
      'ReLU(-1) = ${f1.value}, df/dx at x=-1: ${x1.grad}'); // Expected: 0, grad = 0
  print(
      'ReLU(2) = ${f2.value}, df/dx at x=2: ${x2.grad}'); // Expected: 2, grad = 1

  // Example: Compute derivative of f(x) = sigmoid(x) at x = 0
  Node x3 = Node(0.0);
  Node f3 = x3.sigmoid(); // f(x) = sigmoid(x)
  f3.backward();

  print(
      'Sigmoid(0) = ${f3.value}, df/dx at x=0: ${x3.grad}'); // Expected: 0.5, grad = 0.25

  // Example: Compute derivative of f(x) = tanh(x) at x = 1
  Node x4 = Node(1.0);
  Node f4 = x4.tanh(); // f(x) = tanh(x)
  f4.backward();

  print(
      'Tanh(1) = ${f4.value}, df/dx at x=1: ${x4.grad}'); // Expected: tanh(1), grad = 1 - tanh^2(1)
}
