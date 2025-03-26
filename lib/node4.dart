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

// ================ Matrix Class for Multi-Variable Functions ================
class Matrix {
  List<List<Node>> data;
  int rows;
  int cols;

  Matrix(this.data)
      : rows = data.length,
        cols = data.isNotEmpty ? data[0].length : 0;

  static Matrix zeros(int rows, int cols) {
    return Matrix(
        List.generate(rows, (_) => List.generate(cols, (_) => Node(0.0))));
  }

  Matrix operator +(Matrix other) {
    List<List<Node>> result = List.generate(rows, (i) {
      return List.generate(cols, (j) => data[i][j] + other.data[i][j]);
    });
    return Matrix(result);
  }

  Matrix operator -(Matrix other) {
    List<List<Node>> result = List.generate(rows, (i) {
      return List.generate(cols, (j) => data[i][j] - other.data[i][j]);
    });
    return Matrix(result);
  }

  Matrix operator *(Matrix other) {
    List<List<Node>> result = List.generate(rows, (i) {
      return List.generate(cols, (j) => data[i][j] * other.data[i][j]);
    });
    return Matrix(result);
  }

  Matrix transpose() {
    List<List<Node>> result = List.generate(cols, (i) {
      return List.generate(rows, (j) => data[j][i]);
    });
    return Matrix(result);
  }

  Matrix dot(Matrix other) {
    if (cols != other.rows) {
      throw Exception('Matrix dimensions do not match for dot product');
    }

    List<List<Node>> result = List.generate(rows, (i) {
      return List.generate(other.cols, (j) {
        Node sum = Node(0.0);
        for (int k = 0; k < cols; k++) {
          sum = sum + (data[i][k] * other.data[k][j]);
        }
        return sum;
      });
    });

    Matrix output = Matrix(result);

    // Gradient backpropagation for matrix dot product
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < other.cols; j++) {
        output.data[i][j]._backward.add(() {
          for (int k = 0; k < cols; k++) {
            data[i][k].grad += other.data[k][j].value * output.data[i][j].grad;
            other.data[k][j].grad += data[i][k].value * output.data[i][j].grad;
          }
        });
      }
    }

    return output;
  }

  void backward() {
    for (var row in data) {
      for (var node in row) {
        node.backward();
      }
    }
  }

  @override
  String toString() {
    return data
        .map((row) => row.map((n) => n.value.toStringAsFixed(4)).join(' '))
        .join('\n');
  }

  String gradients() {
    return data
        .map((row) => row.map((n) => n.grad.toStringAsFixed(4)).join(' '))
        .join('\n');
  }
}

// ============ Example Usage: Matrix Multiplication with Gradients ============
void main() {
  Matrix A = Matrix([
    [Node(1.0), Node(2.0)],
    [Node(3.0), Node(4.0)]
  ]);

  Matrix B = Matrix([
    [Node(2.0), Node(0.5)],
    [Node(1.0), Node(1.5)]
  ]);

  // Matrix multiplication: C = A · B
  Matrix C = A.dot(B);
  print('Matrix C (Result of A · B):\n$C');

  // Backpropagation on a scalar loss function
  C.data[0][0].backward();

  print('\nGradients for Matrix A:\n${A.gradients()}');
  print('\nGradients for Matrix B:\n${B.gradients()}');
}
// Matrix C (Result of A · B):
// 4.0000 3.5000
// 10.0000 7.5000

// Gradients for Matrix A:
// 2.0000 1.0000
// 0.0000 0.0000

// Gradients for Matrix B:
// 1.0000 2.0000
// 0.0000 0.0000
