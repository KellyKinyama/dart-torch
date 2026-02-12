import 'dart:math' as math;
import '../tensor/tensor.dart';
import 'transformer_encoder.dart';

void main() {
  print("--- Tensor-Based Transformer Encoder Example ---");

  // 1. Hyperparameters
  const vocabSize = 20;
  const embedSize = 32;
  const blockSize = 10;
  const numLayers = 2;
  const numHeads = 4;
  const learningRate = 0.01;

  // 2. Initialize the Tensor Encoder
  final encoder = TransformerEncoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  // 3. Sample Input: "the cat sat on the mat" -> Token IDs
  final sampleInputSequence = [1, 5, 8, 2, 1, 9];
  print("Input sequence: $sampleInputSequence");

  // 4. Forward Pass
  // Returns a Tensor of shape [SequenceLength, EmbedSize] -> [6, 32]
  final encodedEmbeddings = encoder.forward(sampleInputSequence);

  print("\nEncoded Embeddings (First token, first 5 features):");
  final firstTokenData = encodedEmbeddings.data.sublist(0, 5);
  print(firstTokenData.map((v) => v.toStringAsFixed(4)).toList());
  print("Output Shape: ${encodedEmbeddings.shape}"); // [6, 32]

  // 5. Training Step (Downstream Task Simulation)
  print("\n--- Tensor Training Step ---");

  final target = Tensor.random([1, embedSize]);

  // Use your new slice2D method!
  // It automatically handles children and onBackward.
  // We want 1 row (the first token) and all columns (the features).
  final firstTokenEmbedding = encodedEmbeddings.slice2D(1, embedSize);

  // 6. Calculate Loss and Backpropagate
  final loss = firstTokenEmbedding.mseLoss(target);
  print("Initial Loss: ${loss.data[0].toStringAsFixed(6)}");

  encoder.zeroGrad();
  loss.backward();

  // 7. Optimization (Simple SGD)
  for (var p in encoder.parameters()) {
    for (int i = 0; i < p.data.length; i++) {
      p.data[i] -= learningRate * p.grad[i];
    }
  }

  // 8. Verify Update
  final encodedAfter = encoder.forward(sampleInputSequence);

  // Create a clean slice for verification (no need for children here as we won't backprop)
  final firstTokenAfter = Tensor([1, embedSize]);
  for (int i = 0; i < embedSize; i++) {
    firstTokenAfter.data[i] = encodedAfter.data[i];
  }

  final lossAfter = firstTokenAfter.mseLoss(target);
  print("Loss After 1 Step: ${lossAfter.data[0].toStringAsFixed(6)}");

  if (lossAfter.data[0] < loss.data[0]) {
    print(
        "SUCCESS: Loss decreased. Gradients flowed through the Transformer layers.");
  } else {
    print("FAILURE: Loss did not decrease. Check gradient flow.");
  }
}
