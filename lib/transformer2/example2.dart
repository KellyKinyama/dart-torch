// file: example.dart

import 'transformer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

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
  print("🚀 Starting Transformer Training Example...");

  // 1. --- Model & Optimizer Setup ---
  final vocabSize = 10;
  final embedSize = 16;
  final blockSize = 4; // Context length

  final model = Transformer(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 2,
  );

  final optimizer = SGD(model.parameters(), 0.1);

  // 2. --- Sample Data ---
  // The model will learn to predict the next token in the sequence.
  // For input `[1, 2, 3]`, the target is `[2, 3, 4]`.
  final sampleInputs = [2, 3, 4, 5];
  final sampleTargets = [6, 4, 5, 1]; // The next token for each position

  // 3. --- Training Loop ---
  final epochs = 50;
  print("\nTraining for $epochs epochs...");

  for (int epoch = 0; epoch < epochs; epoch++) {
    // --- Forward Pass ---
    // Get the model's predictions (logits) for each position in the input sequence.
    final logits = model.forward(sampleInputs);

    // --- Loss Calculation ---
    // We use cross-entropy loss, which is standard for classification.
    Value totalLoss = Value(0.0);
    for (int t = 0; t < logits.length; t++) {
      final outputAtT = logits[t];
      final targetAtT = sampleTargets[t];

      // Convert the integer target to a one-hot vector representation.
      final targetVector = ValueVector(List.generate(
        vocabSize,
        (i) => Value(i == targetAtT ? 1.0 : 0.0),
      ));

      // The `crossEntropy` function expects probabilities, so we apply softmax first.
      totalLoss += outputAtT.softmax().crossEntropy(targetVector);
    }

    // Average the loss over the sequence length.
    final meanLoss = totalLoss / Value(logits.length.toDouble());

    // --- Backward Pass & Optimization ---

    // Clear old gradients before the backward pass.
    model.zeroGrad();

    // Compute gradients for all parameters starting from the loss.
    meanLoss.backward();

    // Update the model's weights using the computed gradients.
    optimizer.step();

    if (epoch % 5 == 0 || epoch == epochs - 1) {
      print("Epoch $epoch | Loss: ${meanLoss.data.toStringAsFixed(4)}");
    }
  }

  print("\n✅ Training complete.");

  // 4. --- Inference Example ---
  print("\nRunning inference with a new sequence...");
  final testInputs = [2, 3, 4];
  final finalLogits = model.forward(testInputs);

  // Get the prediction for the very last token
  final lastTokenLogits = finalLogits.last.softmax();

  // Find the token with the highest probability (argmax)
  double maxProb = -1.0;
  int predictedIndex = -1;
  for (int i = 0; i < lastTokenLogits.values.length; i++) {
    if (lastTokenLogits.values[i].data > maxProb) {
      maxProb = lastTokenLogits.values[i].data;
      predictedIndex = i;
    }
  }

  print("Input: $testInputs");
  print(
      "Predicted next token: $predictedIndex (Probability: ${(maxProb * 100).toStringAsFixed(2)}%)");
}
