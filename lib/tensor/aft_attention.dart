import 'linear.dart';
import 'module.dart';
import 'tensor.dart';

class AFTAttention extends Module {
  final Linear key;
  final Linear query;
  final Linear value;
  final Tensor w; // Position bias [MaxSeqLen, MaxSeqLen]
  final int headSize;

  AFTAttention(int embedSize, this.headSize, int maxSeqLen)
      : key = Linear(embedSize, headSize),
        query = Linear(embedSize, headSize),
        value = Linear(embedSize, headSize),
        w = Tensor.random([maxSeqLen, maxSeqLen]);

  // @override
  // Tensor forward(Tensor x) {
  //   // x shape: [T, Dim]
  //   final T = x.shape[0];

  //   // 1. Projections [T, HeadSize]
  //   final k = key.forward(x);
  //   final q = query.forward(x);
  //   final v = value.forward(x);

  //   // 2. AFT-Full formula components
  //   final sigmoidQ = q.sigmoid();
  //   final expK = k.exp();
  //   final expKV = expK * v; // Element-wise product [T, HeadSize]

  //   // 3. Vectorized weighted sums
  //   // We use the position bias matrix 'w' to weight the entire sequence at once.
  //   // numerator = w * (exp(k) * v)  -> [T, T] matmul [T, HeadSize] = [T, HeadSize]
  //   // denominator = w * exp(k)     -> [T, T] matmul [T, HeadSize] = [T, HeadSize]

  //   // We slice 'w' to match the current sequence length T
  //   // (Assuming a simple slice of the first T rows/cols of the bias matrix)
  //   final contextNum = w.matmul(expKV);
  //   final contextDen = w.matmul(expK);

  //   // 4. Final output: Gating * (WeightedValues / WeightSums)
  //   return sigmoidQ * (contextNum / contextDen);
  // }

  @override
  Tensor forward(Tensor x) {
    // x shape: [T, Dim]
    final T = x.shape[0];

    // 1. Projections [T, HeadSize]
    final k = key.forward(x);
    final q = query.forward(x);
    final v = value.forward(x);

    // 2. AFT-Full formula components
    final sigmoidQ = q.sigmoid();
    final expK = k.exp();
    final expKV = expK * v;

    // 3. Vectorized weighted sums
    // FIX: Slice 'w' to match the current sequence length T [T, T]
    // Use the slice2D method to get the first T rows and T columns
    final wSub = w.slice2D(T, T);

    // Now shapes match: [T, T] matmul [T, HeadSize] -> [T, HeadSize]
    final contextNum = wSub.matmul(expKV);
    final contextDen = wSub.matmul(expK);

    // 4. Final output: Gating * (WeightedValues / WeightSums)
    return sigmoidQ * (contextNum / contextDen);
  }

  @override
  List<Tensor> parameters() => [
        ...key.parameters(),
        ...query.parameters(),
        ...value.parameters(),
        w,
      ];
}

void main() {
  // 1. Hyperparameters
  const int sequenceLength = 5; // T
  const int embedSize = 16; // Dim
  const int headSize = 16; // HeadSize
  const int maxSeqLen = 10;
  const double learningRate = 0.01;

  // 2. Initialize Module
  final aft = AFTAttention(embedSize, headSize, maxSeqLen);

  // 3. Create dummy input data [T, Dim]
  // In a real scenario, this would be your word embeddings
  final input = Tensor.random([sequenceLength, embedSize]);

  // 4. Create a target/label Tensor for a simple regression task
  // Let's assume we want the output to match some target values
  final target = Tensor.fill([sequenceLength, headSize], 0.5);

  print('--- Starting AFT Training Step ---');

  // 5. Forward Pass
  final output = aft.forward(input);
  print('Output shape: ${output.shape}'); // Expected: [5, 16]

  // 6. Compute Loss (using MSE logic)
  final loss = output.mseLoss(target);
  print('Initial Loss: ${loss.data[0].toStringAsFixed(6)}');

  // 7. Backward Pass (Autograd)
  // This calculates gradients for weights in Linear layers and the position bias 'w'
  loss.backward();

  // 8. Optimization Step (Simple SGD)
  final params = aft.parameters();
  for (var p in params) {
    for (int i = 0; i < p.length; i++) {
      // Update rule: weight = weight - (learning_rate * gradient)
      p.data[i] -= learningRate * p.grad[i];
    }
    // Clear gradients for the next iteration
    p.zeroGrad();
  }

  // 9. Verify training (Forward pass again)
  final newOutput = aft.forward(input);
  final newLoss = newOutput.mseLoss(target);
  print('Loss after one step: ${newLoss.data[0].toStringAsFixed(6)}');

  if (newLoss.data[0] < loss.data[0]) {
    print('Success: Loss decreased!');
  }
}
