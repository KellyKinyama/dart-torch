// file: cross_attention.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// A single head of cross-attention (Encoder-Decoder Attention).
///
/// This module allows tokens in the decoder's sequence (queries) to interact
/// with and attend to tokens in the encoder's output sequence (keys and values).
class CrossAttention extends Module {
  final Layer key;
  final Layer query;
  final Layer value;
  final int headSize;

  CrossAttention(int decoderEmbedSize, int encoderEmbedSize, this.headSize)
      : key = Layer.fromNeurons(
            encoderEmbedSize, headSize), // Keys from encoder output
        value = Layer.fromNeurons(
            encoderEmbedSize, headSize), // Values from encoder output
        query = Layer.fromNeurons(
            decoderEmbedSize, headSize); // Queries from decoder input

  /// Forward pass for a single cross-attention head.
  ///
  /// Takes a list of decoder token vectors `x_decoder` (for queries) and
  /// a list of encoder output vectors `x_encoder` (for keys and values).
  /// Returns a new list of vectors where each decoder vector is a weighted
  /// sum based on attention scores to the encoder output.
  List<ValueVector> forward(
      List<ValueVector> x_decoder, List<ValueVector> x_encoder) {
    final T_dec = x_decoder.length; // Decoder sequence length
    final T_enc = x_encoder.length; // Encoder sequence length

    // Project input vectors into key, query, and value spaces
    // Queries are from the decoder, Keys and Values from the encoder
    final q =
        x_decoder.map((v) => query.forward(v)).toList(); // (T_dec, headSize)
    final k =
        x_encoder.map((v) => key.forward(v)).toList(); // (T_enc, headSize)
    final v =
        x_encoder.map((v) => value.forward(v)).toList(); // (T_enc, headSize)

    // 1. Compute attention scores ("affinities")
    // Each query from decoder attends to all keys from encoder
    var wei = List.generate(T_dec, (i) {
      final row = List.generate(T_enc, (j) => q[i].dot(k[j]));
      // Scale by 1/sqrt(head_size)
      return ValueVector(row) * Value(1.0 / math.sqrt(headSize.toDouble()));
    });

    // Cross-attention typically does not use masking on the encoder output
    // (decoder can see all encoder output)

    // 2. Apply softmax to get attention weights (probabilities)
    final p_attn = wei.map((row) => row.softmax()).toList();

    // 3. Perform the weighted aggregation of the value vectors
    final out = List.generate(T_dec, (i) {
      var pos_out = ValueVector(List.filled(headSize, Value(0.0)));
      for (int j = 0; j < T_enc; j++) {
        pos_out += v[j] * p_attn[i].values[j]; // Weighted sum
      }
      return pos_out;
    });

    return out;
  }

  @override
  List<Value> parameters() {
    return [...key.parameters(), ...query.parameters(), ...value.parameters()];
  }
}
