import '../auto_diff2.dart';

void main() {
  // Learnable weights
  final A1 = Value(0.1);
  final A2 = Value(0.1);
  final A3 = Value(0.1);
  final A4 = Value(0.1);

  // Training data: Each sample is a List<Value> (features), and each target is a Value
  final samples = [
    [Value(1.0), Value(2.0), Value(3.0), Value(4.0)],
    [Value(2.0), Value(1.0), Value(0.0), Value(1.0)],
    [Value(0.0), Value(1.0), Value(2.0), Value(3.0)],
    [Value(3.0), Value(3.0), Value(3.0), Value(3.0)],
  ];

  final targets = [
    Value(30.0), // target y1
    Value(10.0), // target y2
    Value(20.0), // target y3
    Value(36.0), // target y4
  ];

  const learningRate = 0.01;
  const epochs = 100;

  for (var epoch = 0; epoch < epochs; epoch++) {
    final losses = <Value>[];

    // Clear gradients
    for (var param in [A1, A2, A3, A4]) {
      param.grad = 0;
    }

    // Compute predictions and squared losses
    for (var i = 0; i < samples.length; i++) {
      final x = samples[i];
      final target = targets[i];

      final y = A1 * x[0] + A2 * x[1] + A3 * x[2] + A4 * x[3];
      final diff = y - target;
      final squaredLoss = diff * diff;
      losses.add(squaredLoss);
    }

    // Compute average loss
    Value totalLoss = losses.reduce((a, b) => a + b);
    final avgLoss = totalLoss * (1.0 / samples.length);

    // Backward pass
    avgLoss.backward();

    // Gradient descent
    A1.setData(A1.data - learningRate * A1.grad);
    A2.setData(A2.data - learningRate * A2.grad);
    A3.setData(A3.data - learningRate * A3.grad);
    A4.setData(A4.data - learningRate * A4.grad);

    // Debug output
    print('Epoch $epoch: Loss = ${avgLoss.data.toStringAsFixed(4)}');
    print(
        '  A1=${A1.data.toStringAsFixed(4)}, A2=${A2.data.toStringAsFixed(4)}, '
        'A3=${A3.data.toStringAsFixed(4)}, A4=${A4.data.toStringAsFixed(4)}\n');
  }
}
