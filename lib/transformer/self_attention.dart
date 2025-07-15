// file: self_attention.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// A single head of self-attention.
///
/// This module allows tokens in a sequence to interact with and attend to each other.
class SelfAttention extends Module {
  final Layer key;
  final Layer query;
  final Layer value;
  final int headSize;
  final bool masked;

  SelfAttention(int embedSize, this.headSize, {this.masked = false})
      : key = Layer.fromNeurons(embedSize, headSize),
        query = Layer.fromNeurons(embedSize, headSize),
        value = Layer.fromNeurons(embedSize, headSize);

  /// Forward pass for a single self-attention head.
  ///
  /// It takes a list of token vectors `x` and returns a new list of vectors
  /// where each vector is a weighted sum based on attention scores.
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length; // Sequence length

    // Project input vectors into key, query, and value spaces
    final k = x.map((v) => key.forward(v)).toList(); // (T, headSize)
    final q = x.map((v) => query.forward(v)).toList(); // (T, headSize)
    final v = x.map((v) => value.forward(v)).toList(); // (T, headSize)

    // 1. Compute attention scores ("affinities")
    var wei = List.generate(T, (i) {
      final row = List.generate(T, (j) => q[i].dot(k[j]));
      // Scale by 1/sqrt(head_size)
      return ValueVector(row) * Value(1.0 / math.sqrt(headSize.toDouble()));
    });

    // 2. Apply optional mask (for decoder blocks)
    if (masked) {
      wei = List.generate(T, (i) {
        final newVals = wei[i].values.asMap().entries.map((entry) {
          int j = entry.key;
          Value val = entry.value;
          if (j > i) {
            // Set future tokens to -infinity so they become 0 after softmax
            return Value(double.negativeInfinity, {val}, 'mask');
          }
          return val;
        }).toList();
        return ValueVector(newVals);
      });
    }

    // 3. Apply softmax to get attention weights (probabilities)
    final p_attn = wei.map((row) => row.softmax()).toList();

    // 4. Perform the weighted aggregation of the value vectors
    final out = List.generate(T, (i) {
      var pos_out = ValueVector(List.filled(headSize, Value(0.0)));
      for (int j = 0; j < T; j++) {
        final weighted_v = v[j] * p_attn[i].values[j];
        pos_out += weighted_v;
      }
      return pos_out;
    });

    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...key.parameters(),
      ...query.parameters(),
      ...value.parameters(),
    ];
  }
}

void main() {
  // --- Configuration ---
  // embedSize: The dimensionality of each input token vector.
  // headSize: The dimensionality of the key, query, and value vectors within the attention mechanism.
  // seqLength: The number of tokens in the input sequence.
  const int embedSize = 8;
  const int headSize = 4;
  const int seqLength = 3;

  print('--- Self-Attention Usage Example ---');
  print('Configuration:');
  print('  Embedding Size: $embedSize');
  print('  Head Size:      $headSize');
  print('  Sequence Length: $seqLength');

  // --- Initialize SelfAttention Module ---
  // Create a SelfAttention head. 'masked: false' means it's a standard attention head,
  // not a masked one typically used in decoders for auto-regressive generation.
  final selfAttention = SelfAttention(embedSize, headSize, masked: false);
  print('\nSelfAttention module initialized.');

  // --- Create Dummy Input Sequence ---
  // For this example, we'll create a sequence of random ValueVectors.
  // In a real application, these would be embeddings of actual tokens.
  final List<ValueVector> sequence = List.generate(seqLength, (i) {
    final List<Value> values = List.generate(
        embedSize,
        (j) => Value(math.Random().nextDouble() * 2 -
            1)); // Random values between -1 and 1
    return ValueVector(values);
  });

  print('\n--- INPUT ---');
  print('Input sequence has $seqLength tokens, each with $embedSize features.');
  print('First token (example): ${sequence[0]}');

  // --- Forward Pass ---
  print('\n--- FORWARD PASS ---');
  // Perform the forward pass through the self-attention mechanism.
  // This will compute the attention scores and produce a new sequence
  // where each token is a weighted sum of the input value vectors.
  final outputSequence = selfAttention.forward(sequence);
  print(
      'Output sequence has $seqLength tokens, each with $headSize features (as expected).');
  print('First output token (example): ${outputSequence[0]}');
  print('First output (example): $outputSequence');

  // --- Backward Pass (for training) ---
  print('\n--- BACKWARD PASS ---');
  // To run a backward pass, we need a single scalar loss value.
  // For this example, we'll just sum all the data in the output vectors
  // to create a dummy loss. In a real scenario, this would be a
  // task-specific loss function (e.g., cross-entropy for classification).
  Value totalLoss = Value(0.0);
  for (final vec in outputSequence) {
    for (final val in vec.values) {
      totalLoss += val;
    }
  }
  print('Calculated a dummy total loss: ${totalLoss}');

  // Now, backpropagate from the loss. This will compute gradients for
  // all trainable parameters (weights and biases) within the SelfAttention module.
  totalLoss.backward();
  print('Backward pass complete. Gradients are now calculated.');

  // --- Inspect Gradients ---
  // We can now see the gradients on the parameters of the model.
  // For example, let's look at the first weight of the 'query' layer.
  // Note: The exact gradient values will vary due to random initialization
  // and the dummy loss, but they should be non-zero.
  final firstQueryWeight = selfAttention.query.neurons[0].w.values[0];
  print('\n--- GRADIENT CHECK (Example) ---');
  print('First weight of the query layer: ${firstQueryWeight.data}');
  print('Gradient of the first weight:    ${firstQueryWeight.grad}');

  // You can also zero out gradients before a new training step
  selfAttention.zeroGrad();
  print('\nGradients zeroed out for all parameters.');
  print(
      'First weight of the query layer after zeroGrad: ${firstQueryWeight.grad}');
}
