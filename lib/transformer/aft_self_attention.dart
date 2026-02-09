// file: self_attention.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// Attention Free Transformer (AFT-full) implementation.
///
/// This module replaces standard dot-product attention with a weighted
/// average of values based on keys and learned position biases.
class AFTAttention extends Module {
  final Layer key;
  final Layer query;
  final Layer value;
  final int headSize;
  final bool masked;

  // Learned pair-wise position biases: w_t,t'
  // In AFT-full, this is a TxT matrix of parameters.
  final List<ValueVector> w;

  AFTAttention(int embedSize, this.headSize, int maxSeqLen,
      {this.masked = false})
      : key = Layer.fromNeurons(embedSize, headSize),
        query = Layer.fromNeurons(embedSize, headSize),
        value = Layer.fromNeurons(embedSize, headSize),
        // Initialize position biases with small random values
        w = List.generate(
            maxSeqLen,
            (_) => ValueVector(List.generate(maxSeqLen,
                (_) => Value(math.Random().nextDouble() * 0.02 - 0.01))));

  /// Forward pass for AFT.
  /// Complexity: O(Td) time and O(Td) space (excluding the bias matrix storage).
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length;

    // 1. Project inputs to Q, K, V spaces
    final k = x.map((v) => key.forward(v)).toList(); // (T, headSize)
    final q = x.map((v) => query.forward(v)).toList(); // (T, headSize)
    final v = x.map((v) => value.forward(v)).toList(); // (T, headSize)

    // 2. Apply sigmoid to Query (sigma_q) as per the AFT formula
    final sigmoidQ = q.map((vec) => vec.sigmoid()).toList();

    // 3. Compute the output for each position t
    final out = List.generate(T, (t) {
      // For each dimension, compute:
      // sum(exp(K_t' + w_t,t') * V_t') / sum(exp(K_t' + w_t,t'))

      ValueVector numerator = ValueVector(List.filled(headSize, Value(0.0)));
      ValueVector denominator = ValueVector(List.filled(headSize, Value(0.0)));

      // Range of t' depends on whether the mask is applied
      int endRange = masked ? t + 1 : T;

      for (int tp = 0; tp < endRange; tp++) {
        // Bias for the specific pair of positions (t, t')
        final bias = w[t].values[tp];

        // Compute element-wise exp(K_tp + bias)
        final expKW =
            ValueVector(k[tp].values.map((kv) => (kv + bias).exp()).toList());

        numerator += (expKW * v[tp]); // Element-wise product then sum
        denominator += expKW; // Element-wise sum
      }

      // Normalize the aggregated context
      final context = numerator / denominator;

      // Final output is element-wise product of Gated Query and Context
      return sigmoidQ[t] * context;
    });

    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...key.parameters(),
      ...query.parameters(),
      ...value.parameters(),
      ...w.expand((row) => row.values),
    ];
  }
}

void main() {
  const int embedSize = 8;
  const int headSize = 4;
  const int seqLength = 3;

  print('--- AFT Attention Usage Example ---');

  // Initialize AFT with max sequence length knowledge
  final aft = AFTAttention(embedSize, headSize, seqLength, masked: false);

  final List<ValueVector> sequence = List.generate(seqLength, (i) {
    return ValueVector(List.generate(
        embedSize, (j) => Value(math.Random().nextDouble() * 2 - 1)));
  });

  print('\n--- FORWARD PASS ---');
  final outputSequence = aft.forward(sequence);
  print('Output sequence length: ${outputSequence.length}');
  print('First output token: ${outputSequence[0]}');

  print('\n--- BACKWARD PASS ---');
  Value totalLoss = Value(0.0);
  for (final vec in outputSequence) {
    for (final val in vec.values) {
      totalLoss += val;
    }
  }
  totalLoss.backward();
  print('Gradients calculated for all parameters (including position biases).');

  final firstBiasGrad = aft.w[0].values[0].grad;
  print('Gradient of first position bias (w[0][0]): $firstBiasGrad');
}
