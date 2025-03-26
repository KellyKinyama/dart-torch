import 'dart:math' as math;

class Node {
  double value;
  double grad;
  List<void Function()> _backward = [];
  List<Node> _parents = [];
  String _op; // Operation tracking

  Node(this.value, [this._op = '']) : grad = 0.0;

  Node operator +(Node other) {
    final out = Node(value + other.value, '+');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += out.grad;
      other.grad += out.grad;
    });
    return out;
  }

  Node operator -(Node other) {
    final out = Node(value - other.value, '-');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += out.grad;
      other.grad -= out.grad;
    });
    return out;
  }

  Node operator *(Node other) {
    final out = Node(value * other.value, '*');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += other.value * out.grad;
      other.grad += value * out.grad;
    });
    return out;
  }

  Node operator /(Node other) {
    final out = Node(value / other.value, '/');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += (1 / other.value) * out.grad;
      other.grad -= (value / (other.value * other.value)) * out.grad;
    });
    return out;
  }

  Node pow(double exponent) {
    final out = Node(math.pow(value, exponent).toDouble(), '^$exponent');
    out._parents = [this];
    out._backward.add(() {
      this.grad += (exponent * math.pow(value, exponent - 1)) * out.grad;
    });
    return out;
  }

  Node relu() {
    final out = Node(value > 0 ? value : 0, 'ReLU');
    out._parents = [this];
    out._backward.add(() {
      this.grad += (out.value > 0 ? 1.0 : 0.0) * out.grad;
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
  String toString() => 'Node(value: $value, grad: $grad, op: $_op)';
}

void main() {
  Node x = Node(3.0);
  Node y = Node(4.0);

  Node z = (x * y) + x.pow(2) - y.relu();
  z.backward();

  print('Final result: $z');
  print('Gradient w.r.t x: ${x.grad}');
  print('Gradient w.r.t y: ${y.grad}');
}
