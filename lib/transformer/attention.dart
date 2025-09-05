import 'dart:math' as math;

import 'package:dart_torch/nn/value_vector.dart';

import '../nn/layer.dart';
import '../nn/value.dart';

const int embedSize = 8;
const int headSize = 4;
const int seqLength = 3;

final keys = Layer.fromNeurons(embedSize, headSize);
final queries = Layer.fromNeurons(embedSize, headSize);
final values = Layer.fromNeurons(embedSize, headSize);

List<ValueVector> selfAttentionForward(List<ValueVector> sequence,
    {bool masked = false}) {
  final T = sequence.length;

  final k = sequence.map((key) => keys.forward(key)).toList();
  final q = sequence.map((query) => queries.forward(query)).toList();
  final v = sequence.map((value) => values.forward(value)).toList();

  List<ValueVector> attentionWeights = List.generate(T, (i) {
    final row = List.generate(T, (j) => q[i].dot(k[j]));
    return ValueVector(row) * Value(1.0 / math.sqrt(headSize.toDouble()));
  });

  // 2. Apply optional mask (for decoder blocks)
  if (masked) {
    attentionWeights = List.generate(T, (i) {
      final newVals = attentionWeights[i].values.asMap().entries.map((entry) {
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

  final pAttn = attentionWeights.map((row) => row.softmax()).toList();
  final out = List.generate(T, (i) {
    ValueVector posOut = ValueVector(List.filled(headSize, Value(0.0)));
    for (int j = 0; j < T; j++) {
      final weightedV = v[j] * pAttn[i].values[j];
      posOut += weightedV;
    }
    return posOut;
  });
  return out;
}

void main() {
  // --- Configuration ---
  // embedSize: The dimensionality of each input token vector.
  // headSize: The dimensionality of the key, query, and value vectors within the attention mechanism.
  // seqLength: The number of tokens in the input sequence.

  print('--- Self-Attention Usage Example ---');
  print('Configuration:');
  print('  Embedding Size: $embedSize');
  print('  Head Size:      $headSize');
  print('  Sequence Length: $seqLength');

  // --- Initialize SelfAttention Module ---
  // Create a SelfAttention head. 'masked: false' means it's a standard attention head,
  // not a masked one typically used in decoders for auto-regressive generation.
  // final selfAttention = SelfAttention(embedSize, headSize, masked: false);
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
  final outputSequence = selfAttentionForward(sequence);
  print(
      'Output sequence has $seqLength tokens, each with $headSize features (as expected).');
  // print('First output token (example): ${outputSequence[0]}');
  // print('Second output token (example): ${outputSequence[1]}');
  // print('Third output token (example): ${outputSequence[2]}');
  print('Output (example): $outputSequence');

  // --- Backward Pass (for training) ---
  // print('\n--- BACKWARD PASS ---');
  // To run a backward pass, we need a single scalar loss value.
  // For this example, we'll just sum all the data in the output vectors
  // to create a dummy loss. In a real scenario, this would be a
  // task-specific loss function (e.g., cross-entropy for classification).
  // Value totalLoss = Value(0.0);
  // for (final vec in outputSequence) {
  //   for (final val in vec.values) {
  //     totalLoss += val;
  //   }
  // }
  // print('Calculated a dummy total loss: ${totalLoss}');

  // Now, backpropagate from the loss. This will compute gradients for
  // all trainable parameters (weights and biases) within the SelfAttention module.
  // totalLoss.backward();
  // print('Backward pass complete. Gradients are now calculated.');

  // --- Inspect Gradients ---
  // We can now see the gradients on the parameters of the model.
  // For example, let's look at the first weight of the 'query' layer.
  // Note: The exact gradient values will vary due to random initialization
  // and the dummy loss, but they should be non-zero.
  // final firstQueryWeight = selfAttention.query.neurons[0].w.values[0];
  // print('\n--- GRADIENT CHECK (Example) ---');
  // print('First weight of the query layer: ${firstQueryWeight.data}');
  // print('Gradient of the first weight:    ${firstQueryWeight.grad}');

  // You can also zero out gradients before a new training step
  // selfAttention.zeroGrad();
  // print('\nGradients zeroed out for all parameters.');
  // print(
  //     'First weight of the query layer after zeroGrad: ${firstQueryWeight.grad}');
}
