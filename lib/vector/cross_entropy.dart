import '../auto_diff2.dart';
import 'linear_model2.dart';
import 'value_vector.dart';

void main() {
  final model =
      LinearModel(4, 3, activation: (v) => v.relu(), isClassification: true);

  final inputs = [
    ValueVector([Value(1.0), Value(2.0), Value(3.0), Value(4.0)]),
    ValueVector([Value(2.0), Value(1.0), Value(0.0), Value(1.0)]),
  ];

  final targets = [
    ValueVector([Value(0.0), Value(1.0), Value(0.0)]), // One-hot encoded
    ValueVector([Value(0.0), Value(0.0), Value(1.0)]),
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

      // Cross-Entropy loss (assuming targets are one-hot)
      final diff = yPred - yTrue;
      final loss =
          diff.squared().mean(); // This can be replaced with Cross-Entropy
      // final loss = yPred.crossEntropy(yTrue);
      losses.add(loss);
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
