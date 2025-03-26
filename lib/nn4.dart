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
  List<List<MatrixNode>> weights; // List of weights for each layer
  List<List<MatrixNode>> biases; // List of biases for each layer
  int inputSize;
  int outputSize;
  int hiddenLayerSize;
  int numHiddenLayers;

  NeuralNetwork({
    required this.inputSize,
    required this.outputSize,
    required this.hiddenLayerSize,
    required this.numHiddenLayers,
  })  : weights = List.generate(numHiddenLayers + 1, (_) => []),
        biases = List.generate(numHiddenLayers + 1, (_) => []) {
    // Initialize weights and biases for each layer
    // Input to first hidden layer
    weights[0] = List.generate(
        inputSize,
        (_) => MatrixNode(List.generate(1,
            (_) => List.filled(hiddenLayerSize, math.Random().nextDouble()))));
    biases[0] = List.generate(hiddenLayerSize,
        (_) => MatrixNode(List.generate(1, (_) => List.filled(1, 0.0))));

    // Hidden layers
    for (int i = 1; i < numHiddenLayers; i++) {
      weights[i] = List.generate(
          hiddenLayerSize,
          (_) => MatrixNode(List.generate(
              1,
              (_) =>
                  List.filled(hiddenLayerSize, math.Random().nextDouble()))));
      biases[i] = List.generate(hiddenLayerSize,
          (_) => MatrixNode(List.generate(1, (_) => List.filled(1, 0.0))));
    }

    // Last hidden layer to output layer
    weights[numHiddenLayers] = List.generate(
        hiddenLayerSize,
        (_) => MatrixNode(List.generate(
            1, (_) => List.filled(outputSize, math.Random().nextDouble()))));
    biases[numHiddenLayers] = List.generate(outputSize,
        (_) => MatrixNode(List.generate(1, (_) => List.filled(1, 0.0))));
  }

  MatrixNode forward(List<MatrixNode> inputs) {
    List<MatrixNode> prevLayerOutput = inputs;

    // Pass input through all hidden layers
    for (int layer = 0; layer < numHiddenLayers; layer++) {
      List<MatrixNode> layerOutput = [];
      for (int i = 0; i < prevLayerOutput.length; i++) {
        // Compute each layer's output using weights and biases
        MatrixNode layerInput =
            (prevLayerOutput[i] * weights[layer][i] + biases[layer][i])
                .sigmoid();
        layerOutput.add(layerInput);
      }
      prevLayerOutput = layerOutput;
    }

    // Output layer (Linear Activation)
    MatrixNode output = prevLayerOutput[0] * weights[numHiddenLayers][0] +
        biases[numHiddenLayers][0];
    return output;
  }

  void train(
      List<List<List<double>>> X, List<List<double>> y, int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      MatrixNode loss =
          MatrixNode(List.generate(1, (_) => List.filled(1, 0.0)));
      List<MatrixNode> outputs = [];

      for (int i = 0; i < X.length; i++) {
        List<MatrixNode> inputs = [
          MatrixNode([X[i][0]]),
          MatrixNode([X[i][1]])
        ];
        MatrixNode pred = forward(inputs);
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
      for (var layer = 0; layer < weights.length; layer++) {
        for (int i = 0; i < weights[layer].length; i++) {
          for (int j = 0; j < weights[layer][i].value.length; j++) {
            for (int k = 0; k < weights[layer][i].value[0].length; k++) {
              weights[layer][i].value[j][k] -=
                  lr * weights[layer][i].grad[j][k];
              weights[layer][i].zeroGrad();
            }
          }
        }
      }

      for (var layer = 0; layer < biases.length; layer++) {
        for (int i = 0; i < biases[layer].length; i++) {
          for (int j = 0; j < biases[layer][i].value.length; j++) {
            for (int k = 0; k < biases[layer][i].value[0].length; k++) {
              biases[layer][i].value[j][k] -= lr * biases[layer][i].grad[j][k];
              biases[layer][i].zeroGrad();
            }
          }
        }
      }

      if (epoch % 10 == 0) {
        print('Epoch $epoch - Loss: ${loss.value[0][0]}');
      }
    }
  }
}

// void main() {
//   // Initialize the network with 2 input features, 1 output, 3 hidden layers, and 4 neurons per hidden layer
//   NeuralNetwork nn = NeuralNetwork(
//     inputSize: 2,
//     outputSize: 1,
//     hiddenLayerSize: 4,
//     numHiddenLayers: 3,
//   );

//   // XOR Training Data
//   List<List<List<double>>> X = [
//     [
//       [0.0],
//       [0.0]
//     ],
//     [
//       [0.0],
//       [1.0]
//     ],
//     [
//       [1.0],
//       [0.0]
//     ],
//     [
//       [1.0],
//       [1.0]
//     ]
//   ];
//   List<List<double>> y = [
//     [0.0],
//     [1.0],
//     [1.0],
//     [0.0]
//   ];

//   nn.train(X, y, 1000, 0.02);

//   print('\nFinal Predictions:');
//   for (var x in X) {
//     List<MatrixNode> inputs = [
//       MatrixNode([x[0]]),
//       MatrixNode([x[1]])
//     ];
//     MatrixNode pred = nn.forward(inputs);
//     print('Input: $x -> Prediction: ${pred.value[0][0].toStringAsFixed(4)}');
//   }
// }

void main() {
  // Initialize the network with 2 input features, 1 output, 4 hidden layers, and 4 neurons per hidden layer
  NeuralNetwork nn = NeuralNetwork(
    inputSize: 2, // 2 input features
    outputSize: 1, // 1 output
    hiddenLayerSize: 4, // 4 neurons per hidden layer
    numHiddenLayers: 4, // 4 hidden layers
  );

  // XOR Training Data
  List<List<List<double>>> X = [
    [
      [0.0],
      [0.0]
    ], // Input: (0, 0)
    [
      [0.0],
      [1.0]
    ], // Input: (0, 1)
    [
      [1.0],
      [0.0]
    ], // Input: (1, 0)
    [
      [1.0],
      [1.0]
    ], // Input: (1, 1)
  ];

  List<List<double>> y = [
    [0.0], // Output: 0 for (0, 0)
    [0.0], // Output: 1 for (0, 1)
    [0.0], // Output: 1 for (1, 0)
    [1.0], // Output: 0 for (1, 1)
  ];

  // Train the neural network with 1000 epochs and learning rate of 0.02
  nn.train(X, y, 1000, 0.02);

  print('\nFinal Predictions:');
  // Make predictions for the training data
  for (var x in X) {
    List<MatrixNode> inputs = [
      MatrixNode([x[0]]),
      MatrixNode([x[1]])
    ];
    MatrixNode pred = nn.forward(inputs);
    print('Input: $x -> Prediction: ${pred.value[0][0].toStringAsFixed(4)}');
  }
}
