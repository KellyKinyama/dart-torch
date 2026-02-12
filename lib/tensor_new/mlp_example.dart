// import 'dart:typed_data';
// import 'dart:math' as math;
// Assuming your files are named accordingly
// import 'tensor.dart';
import '../tensor/tensor.dart';
import 'mlp.dart';

void main() {
  // 1. Initialize the MLP
  // Inputs: 2 (for the two binary inputs)
  // Hidden: 4 (enough capacity to learn the XOR non-linearity)
  // Output: 1 (probability/value of the result)
  final model = MLP(2, [4, 1]);
  final double learningRate =
      0.1; // Slightly higher LR for faster XOR convergence

  // 2. Prepare Training Data (XOR Truth Table)
  // Inputs: [0,0], [0,1], [1,0], [1,1]
  final x = Tensor([4, 2]);
  x.data.setAll(0, [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);

  // Targets: 0, 1, 1, 0
  final target = Tensor([4, 1]);
  target.data.setAll(0, [0.0, 1.0, 1.0, 0.0]);

  print('--- Starting Training ---');

  // 3. Training Loop
  for (int epoch = 0; epoch <= 500; epoch++) {
    // Forward pass
    final pred = model.forward(x);
    final loss = pred.mseLoss(target);

    // Backward pass
    model.zeroGrad();
    loss.backward();

    // SGD Update step
    for (var p in model.parameters()) {
      for (int i = 0; i < p.length; i++) {
        p.data[i] -= learningRate * p.grad[i];
      }
    }

    if (epoch % 50 == 0) {
      print("Epoch $epoch | Loss: ${loss.data[0].toStringAsFixed(6)}");
    }
  }

  print('\n--- Final Testing ---');

  // 4. Final Inference and Analysis
  // We run forward one last time to see the trained results
  final test = model.forward(x);

  for (int i = 0; i < 4; i++) {
    var in1 = x.data[i * 2].toInt();
    var in2 = x.data[i * 2 + 1].toInt();
    var prediction = test.data[i];

    // We expect [0,1] and [1,0] to be close to 1.0
    // and [0,0] and [1,1] to be close to 0.0
    print("Input: [$in1, $in2] -> Pred: ${prediction.toStringAsFixed(4)}");
  }
}
