import 'dart:math' as math;

class MatrixNode {
  List<List<double>> value;
  List<List<double>> grad;
  List<void Function()> _backward = [];
  String _op;
  List<MatrixNode> _parents = [];

  MatrixNode(this.value, [this._op = ''])
      : grad = List.generate(
            value.length, (_) => List.filled(value[0].length, 0.0));

  // Matrix Addition
  MatrixNode operator +(MatrixNode other) {
    final outValue = List.generate(
        value.length,
        (i) => List.generate(
            value[0].length, (j) => value[i][j] + other.value[i][j]));
    final out = MatrixNode(outValue, '+');
    out._parents = [this, other];
    out._backward.add(() {
      // Gradients for the addition operation
      for (int i = 0; i < value.length; i++) {
        for (int j = 0; j < value[0].length; j++) {
          grad[i][j] += out.grad[i][j];
          other.grad[i][j] += out.grad[i][j];
        }
      }
    });
    return out;
  }

  // Matrix Subtraction
  MatrixNode operator -(MatrixNode other) {
    final outValue = List.generate(
        value.length,
        (i) => List.generate(
            value[0].length, (j) => value[i][j] - other.value[i][j]));
    final out = MatrixNode(outValue, '-');
    out._parents = [this, other];
    out._backward.add(() {
      // Gradients for the subtraction operation
      for (int i = 0; i < value.length; i++) {
        for (int j = 0; j < value[0].length; j++) {
          grad[i][j] += out.grad[i][j];
          other.grad[i][j] -= out.grad[i][j];
        }
      }
    });
    return out;
  }

  // Matrix Multiplication
  MatrixNode operator *(MatrixNode other) {
    final outValue = List.generate(
        value.length,
        (i) => List.generate(other.value[0].length, (j) {
              double sum = 0.0;
              for (int k = 0; k < value[0].length; k++) {
                sum += value[i][k] * other.value[k][j];
              }
              return sum;
            }));
    final out = MatrixNode(outValue, '*');
    out._parents = [this, other];
    out._backward.add(() {
      // Gradients for the multiplication operation
      for (int i = 0; i < value.length; i++) {
        for (int j = 0; j < other.value[0].length; j++) {
          // Gradients with respect to the first matrix
          for (int k = 0; k < value[0].length; k++) {
            grad[i][k] += out.grad[i][j] * other.value[k][j];
          }
          // Gradients with respect to the second matrix
          for (int k = 0; k < value.length; k++) {
            other.grad[k][j] += value[k][i] * out.grad[i][j];
          }
        }
      }
    });
    return out;
  }

  // Matrix Transpose
  MatrixNode transpose() {
    final outValue = List.generate(value[0].length,
        (i) => List.generate(value.length, (j) => value[j][i]));
    final out = MatrixNode(outValue, 'T');
    out._parents = [this];
    out._backward.add(() {
      // Gradients for the transpose operation
      for (int i = 0; i < value.length; i++) {
        for (int j = 0; j < value[0].length; j++) {
          grad[j][i] += out.grad[i][j];
        }
      }
    });
    return out;
  }

  // Sigmoid Activation Function
  MatrixNode sigmoid() {
    final outValue = List.generate(
        value.length,
        (i) => List.generate(
            value[0].length, (j) => 1 / (1 + math.exp(-value[i][j]))));
    final out = MatrixNode(outValue, 'sigmoid');
    out._parents = [this];
    out._backward.add(() {
      // Gradients for the sigmoid operation
      for (int i = 0; i < value.length; i++) {
        for (int j = 0; j < value[0].length; j++) {
          var sigVal = out.value[i][j];
          grad[i][j] += sigVal * (1 - sigVal) * out.grad[i][j];
        }
      }
    });
    return out;
  }

  // Backward pass to compute gradients
  void backward() {
    grad =
        List.generate(value.length, (_) => List.filled(value[0].length, 0.0));
    grad[0][0] = 1.0; // Example, can be adjusted for specific nodes

    List<MatrixNode> topo = [];
    Set<MatrixNode> visited = {};

    void buildTopo(MatrixNode node) {
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

  void zeroGrad() {
    grad =
        List.generate(value.length, (_) => List.filled(value[0].length, 0.0));
  }

  @override
  String toString() {
    return 'MatrixNode(value: $value, grad: $grad, op: $_op)';
  }
}

class NeuralNetwork {
  List<MatrixNode> weights;
  List<MatrixNode> biases;

  NeuralNetwork()
      : weights = List.generate(
            6,
            (_) => MatrixNode(List.generate(
                2, (_) => List.filled(1, math.Random().nextDouble())))),
        biases = List.generate(
            3, (_) => MatrixNode(List.generate(1, (_) => List.filled(1, 0.0))));

  MatrixNode forward(List<MatrixNode> inputs) {
    MatrixNode h1 =
        (inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0]).sigmoid();
    MatrixNode h2 =
        (inputs[0] * weights[2] + inputs[1] * weights[3] + biases[1]).sigmoid();
    MatrixNode output = (h1 * weights[4] + h2 * weights[5] + biases[2]);
    return output;
  }

  void train(
      List<List<List<double>>> X, List<List<double>> y, int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      MatrixNode loss =
          MatrixNode(List.generate(1, (_) => List.filled(1, 0.0)));
      List<MatrixNode> outputs = [];

      for (int i = 0; i < X.length; i++) {
        // Wrap the inputs in an additional list to make them 2D
        List<MatrixNode> inputs = [
          MatrixNode([X[i][0]]), // Wrap X[i][0] in a list
          MatrixNode([X[i][1]]) // Wrap X[i][1] in a list
        ];
        MatrixNode pred = forward(inputs);

        // Wrap the target in a list to make it 2D
        MatrixNode target = MatrixNode([y[i]]);

        // Mean Squared Error (MSE) Loss
        MatrixNode error = (pred - target);
        error = error * error; // Squared error
        loss = loss + error;
        outputs.add(pred);
      }

      // Compute gradients
      loss.backward();

      // Update weights and biases using gradient descent
      for (var w in weights) {
        for (int i = 0; i < w.value.length; i++) {
          for (int j = 0; j < w.value[0].length; j++) {
            w.value[i][j] -= lr * w.grad[i][j];
            w.zeroGrad();
          }
        }
      }
      for (var b in biases) {
        for (int i = 0; i < b.value.length; i++) {
          for (int j = 0; j < b.value[0].length; j++) {
            b.value[i][j] -= lr * b.grad[i][j];
            b.zeroGrad();
          }
        }
      }

      if (epoch % 10 == 0) {
        print('Epoch $epoch - Loss: ${loss.value[0][0]}');
      }
    }
  }
}

void main() {
  NeuralNetwork nn = NeuralNetwork();

  // Training Data (XOR Problem)
  List<List<List<double>>> X = [
    [
      [0.0],
      [0.0]
    ],
    [
      [0.0],
      [1.0]
    ],
    [
      [1.0],
      [0.0]
    ],
    [
      [1.0],
      [1.0]
    ]
  ];

  List<List<double>> y = [
    [0.0],
    [1.0],
    [1.0],
    [1.0]
  ];

  // Train the neural network
  nn.train(X, y, 2000, 0.05);

  print('\nFinal Predictions:');
  for (var x in X) {
    // Wrap each input (x[0] and x[1]) in an additional list to make them 2D
    List<MatrixNode> inputs = [
      MatrixNode([x[0]]), // Wrap x[0] in a list
      MatrixNode([x[1]]) // Wrap x[1] in a list
    ];
    MatrixNode pred = nn.forward(inputs);
    print('Input: $x -> Prediction: ${pred.value[0][0].toStringAsFixed(4)}');
  }
}
