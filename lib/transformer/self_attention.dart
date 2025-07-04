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
