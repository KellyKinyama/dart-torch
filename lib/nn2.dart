import 'dart:math' as math;

class Node {
  double value;
  double grad;
  List<void Function()> _backward = [];
  List<Node> _parents = [];
  String _op;
  List<List<double>>? jacobian; // To store Jacobian

  Node(this.value, [this._op = '']) : grad = 0.0;

  // Update for operator to handle jacobian calculations
  Node operator +(Node other) {
    final out = Node(value + other.value, '+');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += out.grad;
      other.grad += out.grad;
      if (jacobian != null && other.jacobian != null) {
        for (int i = 0; i < jacobian!.length; i++) {
          for (int j = 0; j < jacobian![i].length; j++) {
            jacobian![i][j] += other.jacobian![i][j];
          }
        }
      }
    });
    return out;
  }

  Node operator -(Node other) {
    final out = Node(value - other.value, '-');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += out.grad;
      other.grad -= out.grad;
      if (jacobian != null && other.jacobian != null) {
        for (int i = 0; i < jacobian!.length; i++) {
          for (int j = 0; j < jacobian![i].length; j++) {
            jacobian![i][j] -= other.jacobian![i][j];
          }
        }
      }
    });
    return out;
  }

  Node operator *(Node other) {
    final out = Node(value * other.value, '*');
    out._parents = [this, other];
    out._backward.add(() {
      this.grad += other.value * out.grad;
      other.grad += value * out.grad;
      if (jacobian != null && other.jacobian != null) {
        for (int i = 0; i < jacobian!.length; i++) {
          for (int j = 0; j < jacobian![i].length; j++) {
            jacobian![i][j] += other.jacobian![i][j] * value;
          }
        }
      }
    });
    return out;
  }

  Node pow(double exponent) {
    final out = Node(math.pow(value, exponent).toDouble(), '^$exponent');
    out._parents = [this];
    out._backward.add(() {
      this.grad += (exponent * math.pow(value, exponent - 1)) * out.grad;
      if (jacobian != null) {
        for (int i = 0; i < jacobian!.length; i++) {
          for (int j = 0; j < jacobian![i].length; j++) {
            jacobian![i][j] += exponent * math.pow(value, exponent - 1);
          }
        }
      }
    });
    return out;
  }

  Node relu() {
    final out = Node(value > 0 ? value : 0, 'ReLU');
    out._parents = [this];
    out._backward.add(() {
      this.grad += (out.value > 0 ? 1.0 : 0.0) * out.grad;
      if (jacobian != null) {
        for (int i = 0; i < jacobian!.length; i++) {
          for (int j = 0; j < jacobian![i].length; j++) {
            jacobian![i][j] += (out.value > 0 ? 1.0 : 0.0);
          }
        }
      }
    });
    return out;
  }

  // Set Jacobian matrix dimensions
  void setJacobian(int outputSize, int inputSize) {
    jacobian = List.generate(outputSize, (i) => List.generate(inputSize, (j) => 0.0));
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

    // Compute gradients for the Jacobian
    for (var node in topo.reversed) {
      for (var backwardOp in node._backward) {
        backwardOp();
      }
    }
  }

  void zeroGrad() {
    grad = 0.0;
  }

  @override
  String toString() =>
      'Node(value: ${value.toStringAsFixed(4)}, grad: ${grad.toStringAsFixed(4)}, op: $_op)';
}

// ========== Neural Network Class ==========

class NeuralNetwork {
  List<Node> weights;
  List<Node> biases;

  NeuralNetwork()
      : weights = List.generate(6, (_) => Node(math.Random().nextDouble())),
        biases = List.generate(3, (_) => Node(0.0)); // 3 Biases for 2 hidden + 1 output neuron

  Node forward(List<Node> inputs) {
    // Hidden Layer (2 Neurons, ReLU Activation)
    Node h1 =
        (inputs[0] * weights[0] + inputs[1] * weights[1] + biases[0]).relu();
    Node h2 =
        (inputs[0] * weights[2] + inputs[1] * weights[3] + biases[1]).relu();

    // Output Layer (Linear Activation)
    Node output = (h1 * weights[4] + h2 * weights[5] + biases[2]);

    return output;
  }

  void train(List<List<double>> X, List<double> y, int epochs, double lr) {
    for (int epoch = 0; epoch < epochs; epoch++) {
      Node loss = Node(0.0);
      List<Node> outputs = [];

      for (int i = 0; i < X.length; i++) {
        List<Node> inputs = [Node(X[i][0]), Node(X[i][1])];
        Node pred = forward(inputs);
        Node target = Node(y[i]);

        // Mean Squared Error (MSE) Loss
        Node error = (pred - target).pow(2);
        loss = loss + error;
        outputs.add(pred);
      }

      // Compute gradients
      loss.backward();

      // Update weights and biases using gradient descent
      for (var w in weights) {
        w.value -= lr * w.grad;
        w.zeroGrad();
      }
      for (var b in biases) {
        b.value -= lr * b.grad;
        b.zeroGrad();
      }

      if (epoch % 10 == 0) {
        print('Epoch $epoch - Loss: ${loss.value}');
      }
    }
  }
}


void extractJacobian(NeuralNetwork nn, List<List<double>> X) {
  for (var x in X) {
    List<Node> inputs = [Node(x[0]), Node(x[1])];
    Node output = nn.forward(inputs);

    // Perform backward pass to calculate gradients and Jacobians
    output.backward();

    // Output the Jacobian of the final output with respect to inputs
    print('Jacobian for input $x:');
    for (int i = 0; i < output.jacobian!.length; i++) {
      print(output.jacobian![i]);
    }
  }
}

// void main() {
//   NeuralNetwork nn = NeuralNetwork();
//   List<List<double>> X = [
//     [0.0, 0.0],
//     [0.0, 1.0],
//     [1.0, 0.0],
//     [1.0, 1.0]
//   ];
  
//   extractJacobian(nn, X);
// }

// ============ Training the Neural Network ============

void main() {
  NeuralNetwork nn = NeuralNetwork();

  // Training Data (XOR Problem)
  List<List<double>> X = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ];
  List<double> y = [0.0, 1.0, 1.0, 0.0]; // XOR labels

  print('Training Neural Network...');
  nn.train(X, y, 1000, 0.02);

  print('\nFinal Predictions:');
  for (var x in X) {
    List<Node> inputs = [Node(x[0]), Node(x[1])];
    Node pred = nn.forward(inputs);
    print('Input: $x -> Prediction: ${pred.value.toStringAsFixed(4)}');
  }
}
