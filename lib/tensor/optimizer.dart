import 'aft_layer.dart';
import 'tensor.dart';

abstract class Optimizer {
  final List<Tensor> parameters;
  Optimizer(this.parameters);

  void step();
  void zeroGrad() {
    for (var p in parameters) {
      p.zeroGrad();
    }
  }
}

class SGD extends Optimizer {
  final double lr;
  final double clipValue; // Add a clip threshold

  SGD(List<Tensor> parameters, {this.lr = 0.01, this.clipValue = 1.0})
      : super(parameters);

  @override
  void step() {
    for (var p in parameters) {
      for (int i = 0; i < p.length; i++) {
        // 1. Clip the gradient to stay between -clipValue and +clipValue
        double g = p.grad[i].clamp(-clipValue, clipValue);

        // 2. Update the weight
        p.data[i] -= lr * g;

        // 3. Emergency NaN check
        if (p.data[i].isNaN) p.data[i] = 0.0;
      }
    }
  }
}

void main() {
  final aft = AFTLayer(16, 10);
  final optimizer =
      SGD(aft.parameters(), lr: 0.1); // Higher LR for visible change

  final x = Tensor.random([5, 16]);
  final target = Tensor.fill([5, 16], 0.5);

  print('--- Training with SGD Optimizer ---');

  // Step 1: Forward
  final output = aft.forward(x);
  final loss = output.mseLoss(target);
  print('Initial Loss: ${loss.data[0].toStringAsFixed(6)}');

  // Step 2: Backward
  optimizer.zeroGrad();
  loss.backward();

  // Step 3: Update
  optimizer.step();

  // Verify
  final nextLoss = aft.forward(x).mseLoss(target);
  print('Loss after Optimizer step: ${nextLoss.data[0].toStringAsFixed(6)}');
}
