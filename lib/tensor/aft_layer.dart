import 'linear.dart';
import 'module.dart';
import 'tensor.dart';

class AFTLayer extends Module {
  final Linear qProj;
  final Linear kProj;
  final Linear vProj;
  final Tensor posBias; // Learned relative position bias

  AFTLayer(int dim, int seqLen)
      : qProj = Linear(dim, dim),
        kProj = Linear(dim, dim),
        vProj = Linear(dim, dim),
        posBias = Tensor.random([seqLen, seqLen]);

  @override
  Tensor forward(Tensor x) {
    // x shape: [T, Dim]
    final T = x.shape[0];

    final Q = qProj.forward(x).sigmoid(); // Gating [T, Dim]
    final K = kProj.forward(x).exp(); // Importance [T, Dim]
    final V = vProj.forward(x); // Information [T, Dim]

    // 1. Slice posBias to match the current sequence length T
    // This creates a [T, T] view of the bias
    final currentBias = posBias.slice2D(T, T);

    // 2. Correct AFT-Full weighting logic
    // We multiply [T, T] (bias) by [T, Dim] (K or K*V)
    // Results in [T, Dim]
    final weight = currentBias.matmul(K);
    final context = currentBias.matmul(K * V);

    // 3. Final Output = Gating * (WeightedValues / WeightSums)
    // Small epsilon (1e-9) added to denominator for numerical stability
    return Q * (context / (weight + 1e-9));
  }

  @override
  List<Tensor> parameters() => [
        ...qProj.parameters(),
        ...kProj.parameters(),
        ...vProj.parameters(),
        posBias
      ];
}

void main() {
  // 1. Define Hyperparameters
  const int dim = 16; // Feature dimension
  const int seqLen = 10; // Maximum sequence length
  const int currentT = 5; // Current input length (must be <= seqLen)
  const double lr = 0.01; // Learning rate

  // 2. Initialize the AFT Layer
  final aft = AFTLayer(dim, seqLen);

  // 3. Prepare Dummy Data
  // Input x: [CurrentT, Dim]
  final x = Tensor.random([currentT, dim]);
  // Target: What we want the network to predict [CurrentT, Dim]
  final target = Tensor.fill([currentT, dim], 0.7);

  print('--- AFTLayer Training Step ---');

  // 4. Forward Pass
  // Note: Inside forward, we must slice posBias to [currentT, currentT]
  // to match the input sequence length.
  final output = aft.forward(x);

  // 5. Calculate Loss (Mean Squared Error)
  final loss = output.mseLoss(target);
  print('Initial Loss: ${loss.data[0].toStringAsFixed(6)}');

  // 6. Backward Pass (Autograd)
  // This populates the .grad field for all parameters (weights, biases, posBias)
  loss.backward();

  // 7. Optimizer Step (Simple Stochastic Gradient Descent)
  final params = aft.parameters();
  for (var p in params) {
    for (int i = 0; i < p.length; i++) {
      p.data[i] -= lr * p.grad[i];
    }
    // Important: Reset gradients for the next training step
    p.zeroGrad();
  }

  // 8. Verify the update
  final nextOutput = aft.forward(x);
  final nextLoss = nextOutput.mseLoss(target);
  print('Loss after 1 step: ${nextLoss.data[0].toStringAsFixed(6)}');

  if (nextLoss.data[0] < loss.data[0]) {
    print('Success: The model is learning!');
  }
}
