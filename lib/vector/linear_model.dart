import '../auto_diff2.dart';
import 'value_vector.dart';

class LinearModel {
  final int inputSize;
  final int outputSize;
  late List<ValueVector> weights; // One per output neuron
  late List<Value> biases; // One per output

  LinearModel(this.inputSize, this.outputSize) {
    weights = List.generate(
        outputSize,
        (_) =>
            ValueVector(List.generate(inputSize, (_) => Value(_randWeight()))));
    biases = List.generate(outputSize, (_) => Value(0.0));
  }

  // Forward pass for one input vector
  ValueVector predict(ValueVector input) {
    final outputs = List<Value>.generate(outputSize, (i) {
      return weights[i].dot(input) + biases[i];
    });
    return ValueVector(outputs);
  }

  // All trainable parameters
  List<Value> parameters() => weights.expand((v) => v.values).toList() + biases;

  double _randWeight() =>
      (0.5 - (DateTime.now().microsecondsSinceEpoch % 1000) / 1000.0);
}

void main() {
  final model = LinearModel(4, 2); // 4 inputs → 2 outputs

  final inputs = [
    ValueVector([Value(1.0), Value(2.0), Value(3.0), Value(4.0)]),
    ValueVector([Value(2.0), Value(1.0), Value(0.0), Value(1.0)]),
    ValueVector([Value(0.0), Value(1.0), Value(2.0), Value(3.0)]),
    ValueVector([Value(3.0), Value(3.0), Value(3.0), Value(3.0)]),
  ];

  final targets = [
    ValueVector([Value(30.0), Value(5.0)]),
    ValueVector([Value(10.0), Value(3.0)]),
    ValueVector([Value(20.0), Value(4.0)]),
    ValueVector([Value(36.0), Value(8.0)]),
  ];

  const epochs = 100;
  const lr = 0.01;

  for (int epoch = 0; epoch < epochs; epoch++) {
    final losses = <Value>[];

    // Reset gradients
    for (var p in model.parameters()) {
      p.grad = 0;
    }

    // Compute loss for all samples
    for (int i = 0; i < inputs.length; i++) {
      final yPred = model.predict(inputs[i]);
      final yTrue = targets[i];
      final diff = yPred - yTrue;
      final squared = diff.squared();
      final sampleLoss = squared.mean();
      losses.add(sampleLoss);
    }

    final totalLoss = losses.reduce((a, b) => a + b);
    final avgLoss = totalLoss * (1.0 / inputs.length);
    avgLoss.backward();

    // Gradient descent
    for (var p in model.parameters()) {
      p.setData(p.data - lr * p.grad);
    }

    print("Epoch $epoch | Loss = ${avgLoss.data.toStringAsFixed(4)}");
  }
}
