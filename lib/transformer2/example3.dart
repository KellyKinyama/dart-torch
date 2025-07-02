import 'dart:math';
import 'transformer.dart';
import 'self_attention.dart';
import 'multi_head_attention.dart';
import 'feed_forward.dart';
import 'layer_norm2.dart'; // Using the more concise LayerNorm
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';
import 'value_matrix.dart'; // For matrix operations if needed

/// A simple Stochastic Gradient Descent (SGD) optimizer.
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  /// Updates each parameter using its calculated gradient.
  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("--- Advanced Transformer Examples ---");

  // Example 1: Inspecting a single SelfAttention head
  exampleSelfAttention();

  // Example 2: Verifying MultiHeadAttention output shape
  exampleMultiHeadAttention();

  // Example 3: Demonstrating Layer Normalization
  exampleLayerNorm();

  // Example 4: Using ValueMatrix for a custom operation (if needed)
  exampleValueMatrix();

  // Example 5: Training a Transformer with a larger vocabulary and sequence
  exampleLargerTransformerTraining();

  // Example 6: Generating a sequence (simplified, as full text generation is complex)
  exampleSequenceGeneration();
}

void exampleSelfAttention() {
  print("\n--- Example 1: SelfAttention Inspection ---");

  final embedSize = 8;
  final headSize = 4;
  final sequenceLength = 3;

  final sa = SelfAttention(embedSize, headSize, masked: false);

  // Create dummy input sequence (e.g., 3 tokens, each with embedSize features)
  final x = List.generate(
      sequenceLength,
      (i) => ValueVector.fromDoubleList(
          List.generate(embedSize, (j) => Random().nextDouble())));

  print(
      "Input to SelfAttention (first token): ${x[0].values.map((v) => v.data.toStringAsFixed(2)).toList()}");

  final output = sa.forward(x);

  print(
      "Output from SelfAttention (first token): ${output[0].values.map((v) => v.data.toStringAsFixed(2)).toList()}");
  print("SelfAttention parameters count: ${sa.parameters().length}");
}

void exampleMultiHeadAttention() {
  print("\n--- Example 2: MultiHeadAttention Shape Verification ---");

  final embedSize = 16;
  final numHeads = 4;
  final sequenceLength = 5;

  final mha = MultiHeadAttention(numHeads, embedSize, masked: true);

  final x = List.generate(
      sequenceLength,
      (i) => ValueVector.fromDoubleList(
          List.generate(embedSize, (j) => Random().nextDouble())));

  print("Input sequence length: ${x.length}");
  print("Input embedding size: ${x[0].values.length}");

  final output = mha.forward(x);

  print("Output sequence length: ${output.length}");
  print("Output embedding size: ${output[0].values.length}");
  assert(output.length == sequenceLength);
  assert(output[0].values.length == embedSize);
  print(
      "MultiHeadAttention output shape is correct: ($sequenceLength, $embedSize)");
  print("MultiHeadAttention parameters count: ${mha.parameters().length}");
}

void exampleLayerNorm() {
  print("\n--- Example 3: Layer Normalization Demonstration ---");

  final dim = 5;
  final ln = LayerNorm(dim);

  // Example vector that is not normalized
  final inputVector = ValueVector([
    Value(10.0),
    Value(20.0),
    Value(30.0),
    Value(40.0),
    Value(50.0),
  ]);

  print(
      "Input vector: ${inputVector.values.map((v) => v.data.toStringAsFixed(2)).toList()}");

  final normalizedVector = ln.forward(inputVector);

  print(
      "Normalized vector: ${normalizedVector.values.map((v) => v.data.toStringAsFixed(2)).toList()}");

  // To check if it's "normalized" (mean ~0, variance ~1) before gamma/beta:
  // You would need to temporarily set gamma=1, beta=0, and then calculate mean/variance of normalizedVector's data.
  // For simplicity, we just observe the output values.
  print("LayerNorm parameters count: ${ln.parameters().length}");
}

void exampleValueMatrix() {
  print("\n--- Example 4: ValueMatrix Usage ---");

  // Create two ValueMatrices
  final matrixA = ValueMatrix([
    [Value(1.0), Value(2.0)],
    [Value(3.0), Value(4.0)]
  ]);

  final matrixB = ValueMatrix([
    [Value(5.0), Value(6.0)],
    [Value(7.0), Value(8.0)]
  ]);

  print("Matrix A:\n$matrixA");
  print("Matrix B:\n$matrixB");

  // Matrix multiplication
  final product = matrixA.multiply(matrixB);
  print("A * B:\n$product");

  // Transpose
  final transposedA = matrixA.transpose();
  print("Transpose of A:\n$transposedA");

  // Scalar addition
  final scalarAdd = matrixA + Value(10.0);
  print("A + 10:\n$scalarAdd");

  // Matrix addition
  final matrixAdd = matrixA + matrixB;
  print("A + B:\n$matrixAdd");

  // Scalar multiplication
  final scalarMul = matrixA * Value(2.0);
  print("A * 2:\n$scalarMul");

  // Applying activation
  final reluA = matrixA.relu();
  print("ReLU(A):\n$reluA");
}

void exampleLargerTransformerTraining() {
  print(
      "\n--- Example 5: Training a Transformer with a larger vocabulary and sequence ---");

  final vocabSize = 50; // Increased vocabulary
  final embedSize = 32;
  final blockSize = 8; // Longer context
  final numLayers = 3;
  final numHeads = 4;

  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  final optimizer =
      SGD(model.parameters(), 0.05); // Slightly reduced learning rate

  // More complex sample data
  final sampleInputs = [0, 1, 5, 2, 8, 12, 3, 10]; // 8 tokens
  final sampleTargets = [1, 5, 2, 8, 12, 3, 10, 15]; // Next tokens for each

  final epochs = 100;
  print("\nTraining for $epochs epochs with larger data...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    final logits = model.forward(sampleInputs);

    Value totalLoss = Value(0.0);
    for (int t = 0; t < logits.length; t++) {
      final outputAtT = logits[t];
      final targetAtT = sampleTargets[t];

      final targetVector = ValueVector(List.generate(
        vocabSize,
        (i) => Value(i == targetAtT ? 1.0 : 0.0),
      ));
      totalLoss += outputAtT.softmax().crossEntropy(targetVector);
    }

    final meanLoss = totalLoss / Value(logits.length.toDouble());

    model.zeroGrad();
    meanLoss.backward();
    optimizer.step();

    if (epoch % 10 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${meanLoss.data.toStringAsFixed(4)}");
    }
  }
  print("✅ Larger model training complete.");
}

void exampleSequenceGeneration() {
  print("\n--- Example 6: Simplified Sequence Generation ---");

  // This is a very basic generative example. True generation involves
  // sampling from predicted probabilities and feeding the sampled token back.
  // The current model is decoder-only, so it can do this.

  final vocabSize = 10;
  final embedSize = 16;
  final blockSize = 4;

  // We'll load a pre-trained (or simply initialized) model
  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 2,
  );
  // In a real scenario, you'd load trained weights here.
  // For this example, we'll just use the randomly initialized model.

  List<int> prompt = [1, 2]; // Start with tokens 1, 2
  final int maxNewTokens = 5;

  print("Prompt: $prompt");
  print("Generating $maxNewTokens new tokens...");

  List<int> generatedSequence = List.from(prompt);

  for (int i = 0; i < maxNewTokens; i++) {
    // Crop the sequence to the block size if it exceeds it
    final currentInput = generatedSequence.length > blockSize
        ? generatedSequence.sublist(generatedSequence.length - blockSize)
        : generatedSequence;

    // Forward pass to get logits
    final logits = model.forward(currentInput);

    // Get the logits for the last token in the sequence (which is the prediction for the *next* token)
    final lastTokenLogits = logits.last;

    // Apply softmax to get probabilities
    final probabilities = lastTokenLogits.softmax();

    // Find the token with the highest probability (greedy sampling)
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    // Add the predicted token to the sequence
    generatedSequence.add(predictedNextToken);
    print(
        "Step ${i + 1}: Predicted token: $predictedNextToken (Prob: ${(maxProb * 100).toStringAsFixed(2)}%)");
  }

  print("Generated sequence: $generatedSequence");
  print(
      "Note: This is a simplified example. For better generation, consider techniques like top-k or nucleus sampling.");
}
