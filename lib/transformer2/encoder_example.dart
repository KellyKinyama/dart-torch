// main.dart (or a new file like example_encoder.dart)

import 'transformer_encoder.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'dart:math';

// Assuming SGD is defined as in your example.dart
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }
}

void main() {
  print("--- Transformer Encoder Example ---");

  final vocabSize = 20;
  final embedSize = 32;
  final blockSize = 10;
  final numLayers = 2;
  final numHeads = 4;

  // Initialize the Transformer Encoder model
  final encoder = TransformerEncoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  // You would typically define a custom loss function and optimizer for encoder tasks
  // For demonstration, let's just show a forward pass.
  // In a real scenario, you might have a downstream task (e.g., classification)
  // that provides a loss to backpropagate through the encoder.

  // Sample input sequence (e.g., token IDs)
  // Let's say input is a sentence like "the cat sat on the mat"
  // and we map words to integer IDs: [1, 5, 8, 2, 1, 9]
  final sampleInputSequence = [1, 5, 8, 2, 1, 9];
  print("Input sequence: $sampleInputSequence");

  // Perform a forward pass to get contextualized embeddings
  final encodedEmbeddings = encoder.forward(sampleInputSequence);

  print(
      "\nEncoded Embeddings (first token's data - only showing first 5 values):");
  if (encodedEmbeddings.isNotEmpty) {
    print(encodedEmbeddings[0]
        .values
        .sublist(0, min(5, encodedEmbeddings[0].values.length))
        .map((v) => v.data.toStringAsFixed(4))
        .toList());
    print(
        "Shape of encoded embeddings: (${encodedEmbeddings.length}, ${encodedEmbeddings[0].values.length})");
    assert(encodedEmbeddings.length == sampleInputSequence.length);
    assert(encodedEmbeddings[0].values.length == embedSize);
    print("Encoder output shape is correct.");
  } else {
    print("No encoded embeddings produced.");
  }

  // To demonstrate backpropagation (e.g., for a classification task)
  // Let's assume we want the first token's embedding to be close to some target vector
  if (encodedEmbeddings.isNotEmpty) {
    print("\n--- Dummy Training Step for Encoder ---");
    // Dummy target for the first token's embedding (e.g., for a classification head attached to it)
    final dummyTarget = ValueVector.fromDoubleList(
        List.generate(embedSize, (i) => Random().nextDouble() * 2 - 1));

    // Simple mean squared error loss for demonstration
    Value dummyLoss = Value(0.0);
    for (int i = 0; i < embedSize; i++) {
      dummyLoss +=
          (encodedEmbeddings[0].values[i] - dummyTarget.values[i]).pow(2);
    }
    dummyLoss = dummyLoss / Value(embedSize.toDouble()); // Mean squared error

    print("Initial Dummy Loss: ${dummyLoss.data.toStringAsFixed(4)}");

    // Zero gradients, backward pass, and optimizer step
    encoder.zeroGrad();
    dummyLoss.backward();

    final optimizer = SGD(encoder.parameters(), 0.01);
    optimizer.step();

    // Re-run forward pass to see the effect of the update
    final encodedEmbeddingsAfterUpdate = encoder.forward(sampleInputSequence);
    Value dummyLossAfterUpdate = Value(0.0);
    for (int i = 0; i < embedSize; i++) {
      dummyLossAfterUpdate +=
          (encodedEmbeddingsAfterUpdate[0].values[i] - dummyTarget.values[i])
              .pow(2);
    }
    dummyLossAfterUpdate = dummyLossAfterUpdate / Value(embedSize.toDouble());

    print(
        "Dummy Loss After 1 Step: ${dummyLossAfterUpdate.data.toStringAsFixed(4)}");
    print(
        "This shows the encoder's parameters are updated based on a downstream loss.");
  }
}
