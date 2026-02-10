import 'linear.dart';
import 'module.dart';
import 'tensor.dart';

class MLP extends Module {
  final List<Linear> layers = [];

  MLP(int nin, List<int> nouts) {
    int currentIn = nin;
    for (var nout in nouts) {
      layers.add(Linear(currentIn, nout));
      currentIn = nout;
    }
  }

  Tensor forward(Tensor x) {
    Tensor out = x;
    for (int i = 0; i < layers.length; i++) {
      out = layers[i].forward(out);
      // Apply ReLU activation to all but the last layer
      if (i < layers.length - 1) {
        out = out.relu();
      }
    }
    return out;
  }

  @override
  List<Tensor> parameters() {
    return layers.expand((l) => l.parameters()).toList();
  }
}

void main() {
  // 1. Define the MLP Architecture
  // Input: 3 features (e.g., Temperature, Humidity, Pressure)
  // Hidden Layer: 10 neurons
  // Output Layer: 1 neuron (e.g., Probability of Rain)
  final model = MLP(3, [10, 1]);

  // 2. Mock Training Data (Batch of 4 samples)
  final x = Tensor([4, 3]);
  x.data.setAll(0, [
    1.0, 0.5, 0.2, // Sample 1
    -1.0, 0.0, 1.0, // Sample 2
    0.5, -0.5, 0.8, // Sample 3
    0.1, 0.1, 0.1 // Sample 4
  ]);

  final target = Tensor([4, 1]);
  target.data.setAll(0, [1.0, 0.0, 1.0, 0.0]); // Ground truth labels

  print("--- Starting Training Loop ---");

  double learningRate = 0.01;

  for (int epoch = 0; epoch <= 100; epoch++) {
    // A. Forward Pass
    // The MLP automatically handles ReLU on hidden layers
    final rawOutput = model.forward(x);
    final prediction = rawOutput.sigmoid(); // Squash to 0-1 probability

    // B. Compute Loss
    final loss = prediction.mseLoss(target);

    // C. Backward Pass
    model.zeroGrad(); // Clears all weight/bias gradients in one call
    loss.backward(); // Propagates error back through the whole MLP

    // D. Optimization (SGD)
    // We iterate through every weight and bias tensor in the model
    final params = model.parameters();
    for (var p in params) {
      for (int i = 0; i < p.length; i++) {
        p.data[i] -= learningRate * p.grad[i];
      }
    }

    if (epoch % 20 == 0) {
      print("Epoch $epoch | Loss: ${loss.data[0].toStringAsFixed(6)}");
    }
  }

  print("\nFinal Predictions:");
  final finalPred = model.forward(x).sigmoid();
  for (int i = 0; i < 4; i++) {
    print(
        "Sample $i: Target ${target.data[i]} -> Pred ${finalPred.data[i].toStringAsFixed(4)}");
  }
}
