import 'dart:math' as math;

class Node {
  double value;
  double grad;
  List<void Function()> _backward = [];
  List<Node> _parents = [];

  Node(this.value) : grad = 0.0;

  // Multiplication with autograd
  Node operator *(Node other) {
    final out = Node(value * other.value);
    out._parents = [this, other];
    out._backward.add(() {
      grad += other.value * out.grad;
      other.grad += value * out.grad;
    });
    return out;
  }

  // Addition with autograd
  Node operator +(Node other) {
    final out = Node(value + other.value);
    out._parents = [this, other];
    out._backward.add(() {
      grad += out.grad;
      other.grad += out.grad;
    });
    return out;
  }

  // Subtraction with autograd
  Node operator -(Node other) {
    final out = Node(value - other.value);
    out._parents = [this, other];
    out._backward.add(() {
      grad += out.grad;
      other.grad -= out.grad;
    });
    return out;
  }

  // Division with autograd
  Node operator /(Node other) {
    final out = Node(value / other.value);
    out._parents = [this, other];
    out._backward.add(() {
      grad += (1 / other.value) * out.grad;
      other.grad -= (value / (other.value * other.value)) * out.grad;
    });
    return out;
  }

  // Exponential function
  Node exp() {
    final expVal = math.exp(value);
    final out = Node(expVal);
    out._parents = [this];
    out._backward.add(() {
      grad += expVal * out.grad;
    });
    return out;
  }

  // Compute gradients via backpropagation
  void backward() {
    grad = 1.0; // Start from final output

    List<Node> topo = [];
    Set<Node> visited = {};

    // Build a topological order of the computational graph
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

    // Traverse graph in reverse order and apply gradients
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
  // Example: Compute derivative of f(x, y) = x^2 + y^2 at (x=3, y=4)
  Node x = Node(3.0);
  Node y = Node(4.0);
  Node f = (x * x) + (y * y); // f(x, y) = x^2 + y^2

  f.backward();

  print('f(x, y) at (3,4): ${f.value}'); // Expected: 3^2 + 4^2 = 25
  print('df/dx at x=3: ${x.grad}'); // Expected: 2(3) = 6
  print('df/dy at y=4: ${y.grad}'); // Expected: 2(4) = 8
}
