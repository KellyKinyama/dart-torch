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

// ================= Matrix Class for Multi-Variable Functions ===================
class Matrix {
  List<List<Node>> data;
  int rows;
  int cols;

  Matrix(this.data)
      : rows = data.length,
        cols = data.isNotEmpty ? data[0].length : 0;

  static Matrix identity(int size) {
    return Matrix(List.generate(size, (i) {
      return List.generate(size, (j) => Node(i == j ? 1.0 : 0.0));
    }));
  }

  Matrix operator +(Matrix other) {
    return Matrix(List.generate(rows, (i) {
      return List.generate(cols, (j) => data[i][j] + other.data[i][j]);
    }));
  }

  Matrix operator -(Matrix other) {
    return Matrix(List.generate(rows, (i) {
      return List.generate(cols, (j) => data[i][j] - other.data[i][j]);
    }));
  }

  Matrix operator *(Matrix other) {
    return Matrix(List.generate(rows, (i) {
      return List.generate(cols, (j) => data[i][j] * other.data[i][j]);
    }));
  }

  Matrix transpose() {
    return Matrix(List.generate(cols, (i) {
      return List.generate(rows, (j) => data[j][i]);
    }));
  }

  Matrix dot(Matrix other) {
    if (cols != other.rows)
      throw Exception('Matrix dimensions do not match for dot product');

    return Matrix(List.generate(rows, (i) {
      return List.generate(other.cols, (j) {
        Node sum = Node(0.0);
        for (int k = 0; k < cols; k++) {
          sum = sum + (data[i][k] * other.data[k][j]);
        }
        return sum;
      });
    }));
  }

  Node determinant() {
    if (rows != cols)
      throw Exception('Determinant only defined for square matrices');

    if (rows == 2) {
      // Base case for 2x2 matrix
      return (data[0][0] * data[1][1]) - (data[0][1] * data[1][0]);
    }

    // Recursive determinant calculation (Laplace expansion)
    Node det = Node(0.0);
    for (int i = 0; i < cols; i++) {
      Matrix subMatrix = _minorMatrix(0, i);
      Node cofactor = data[0][i] * subMatrix.determinant();
      if (i % 2 == 1) cofactor = cofactor * Node(-1.0);
      det = det + cofactor;
    }

    return det;
  }

  Matrix inverse() {
    if (rows != cols)
      throw Exception('Matrix inversion only defined for square matrices');

    Node det = determinant();
    if (det.value == 0.0)
      throw Exception('Matrix is singular, cannot be inverted');

    // Compute adjugate (cofactor matrix)
    Matrix cofactorMatrix = _cofactorMatrix();
    Matrix adjugate = cofactorMatrix.transpose();

    // Multiply adjugate by 1/det(A)
    Matrix inv = adjugate *
        Matrix(List.generate(
            rows, (_) => List.generate(cols, (_) => Node(1.0 / det.value))));

    return inv;
  }

  Matrix _minorMatrix(int row, int col) {
    List<List<Node>> minorData = [];
    for (int i = 0; i < rows; i++) {
      if (i == row) continue;
      List<Node> minorRow = [];
      for (int j = 0; j < cols; j++) {
        if (j == col) continue;
        minorRow.add(data[i][j]);
      }
      minorData.add(minorRow);
    }
    return Matrix(minorData);
  }

  Matrix _cofactorMatrix() {
    List<List<Node>> cofactors = List.generate(rows, (i) {
      return List.generate(cols, (j) {
        Node cofactor = _minorMatrix(i, j).determinant();
        return (i + j) % 2 == 0 ? cofactor : cofactor * Node(-1.0);
      });
    });
    return Matrix(cofactors);
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

// ============ Example Usage: Determinant & Inverse ============
void main() {
  Matrix A = Matrix([
    [Node(4.0), Node(3.0)],
    [Node(3.0), Node(2.0)]
  ]);

  print('Determinant of A: ${A.determinant()}');

  Matrix invA = A.inverse();
  print('\nInverse of A:\n$invA');

  invA.data[0][0].backward();
  print('\nGradients of A:\n${A.gradients()}');
}
