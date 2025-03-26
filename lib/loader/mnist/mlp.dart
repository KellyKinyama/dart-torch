import 'dart:math';
import 'dart:typed_data';

class MultiLayerPerceptron {
  final int inputSize;
  final List<int> hiddenSizes;
  final int outputSize;
  final double learningRate;

  late List<Float32List> weights;
  late List<Float32List> biases;

  MultiLayerPerceptron({
    required this.inputSize,
    required this.hiddenSizes,
    required this.outputSize,
    this.learningRate = 0.01,
  }) {
    _initializeWeights();
  }

  void _initializeWeights() {
    final sizes = [inputSize, ...hiddenSizes, outputSize];
    final random = Random();

    weights = List.generate(sizes.length - 1, (i) {
      return Float32List.fromList(List.generate(sizes[i] * sizes[i + 1], (j) => random.nextDouble() - 0.5));
    });

    biases = List.generate(sizes.length - 1, (i) {
      return Float32List.fromList(List.generate(sizes[i + 1], (j) => 0.0));
    });
  }

  Float32List forward(Float32List input) {
    Float32List activations = input;

    for (int i = 0; i < weights.length; i++) {
      final newActivations = Float32List(biases[i].length);

      for (int j = 0; j < biases[i].length; j++) {
        double sum = biases[i][j];

        for (int k = 0; k < activations.length; k++) {
          sum += activations[k] * weights[i][k * biases[i].length + j];
        }

        newActivations[j] = _relu(sum);
      }

      activations = newActivations;
    }

    return _softmax(activations);
  }

  double train(Float32List input, List<double> target) {
    final output = forward(input);
    final loss = _crossEntropyLoss(output, target);
    return loss; // Backpropagation (not implemented in this snippet)
  }

  List<double> predict(Float32List input) {
    return forward(input);
  }

  static double _relu(double x) => x > 0 ? x : 0;
  static List<double> _softmax(Float32List values) {
    final expValues = values.map((v) => exp(v)).toList();
    final sumExp = expValues.reduce((a, b) => a + b);
    return expValues.map((v) => v / sumExp).toList();
  }

  static double _crossEntropyLoss(List<double> output, List<double> target) {
    return -target.asMap().entries.map((e) => e.value * log(output[e.key] + 1e-9)).reduce((a, b) => a + b);
  }
}
