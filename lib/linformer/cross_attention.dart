// file: cross_attention.dart

// import '../transformer/cross_attention.dart';
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';
import 'dart:math' as math;

/// A single cross-attention head.
class CrossAttention extends Module {
  final Layer query;
  final Layer key;
  final Layer value;
  final int headSize;

  CrossAttention(int decoderEmbedSize, int encoderEmbedSize, this.headSize)
      : query = Layer.fromNeurons(decoderEmbedSize, headSize),
        key = Layer.fromNeurons(encoderEmbedSize, headSize),
        value = Layer.fromNeurons(encoderEmbedSize, headSize);

  /// Forward pass for a single cross-attention head.
  List<ValueVector> forward(
      List<ValueVector> xDecoder, List<ValueVector> xEncoder) {
    final T_dec = xDecoder.length;
    final T_enc = xEncoder.length;

    final q = xDecoder.map((v) => query.forward(v)).toList();
    final k = xEncoder.map((v) => key.forward(v)).toList();
    final v = xEncoder.map((v) => value.forward(v)).toList();

    final List<List<Value>> attention = List.generate(T_dec, (i) {
      return List.generate(T_enc, (j) {
        final score = q[i].dot(k[j]);
        return score / math.sqrt(headSize);
      });
    });

    final weights = attention.map((row) => ValueVector(row).softmax()).toList();

    final List<ValueVector> out = List.generate(T_dec, (i) {
      final List<Value> outputValues = List.generate(headSize, (j) {
        Value sum = Value(0.0);
        for (int t = 0; t < T_enc; t++) {
          sum += weights[i].values[t] * v[t].values[j];
        }
        return sum;
      });
      return ValueVector(outputValues);
    });

    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...query.parameters(),
      ...key.parameters(),
      ...value.parameters(),
    ];
  }
}

/// Linformer variant of a single cross-attention head.
class LinformerCrossAttention extends CrossAttention {
  final Layer keyProj;
  final Layer valueProj;
  final int projK;

  LinformerCrossAttention(
      int decoderEmbedSize, int encoderEmbedSize, int headSize,
      {required this.projK})
      : keyProj = Layer.fromNeurons(encoderEmbedSize, projK),
        valueProj = Layer.fromNeurons(encoderEmbedSize, projK),
        super(decoderEmbedSize, encoderEmbedSize, headSize);

  @override
  List<ValueVector> forward(
      List<ValueVector> xDecoder, List<ValueVector> xEncoder) {
    final T_dec = xDecoder.length;
    final T_enc = xEncoder.length;

    // 1. Linearly project the keys and values from the encoder output to a smaller sequence length (k)
    final projectedKeys = xEncoder.map((v) => keyProj.forward(v)).toList();
    final projectedValues = xEncoder.map((v) => valueProj.forward(v)).toList();
    final q = xDecoder.map((v) => query.forward(v)).toList();

    // 2. Calculate attention with the projected key and value
    final List<List<Value>> attention = List.generate(T_dec, (i) {
      return List.generate(T_enc, (j) {
        final score = q[i].dot(projectedKeys[j]);
        return score / math.sqrt(headSize);
      });
    });

    final weights = attention.map((row) => ValueVector(row).softmax()).toList();

    final List<ValueVector> out = List.generate(T_dec, (i) {
      final List<Value> outputValues = List.generate(headSize, (j) {
        Value sum = Value(0.0);
        for (int t = 0; t < T_enc; t++) {
          sum += weights[i].values[t] * projectedValues[t].values[j];
        }
        return sum;
      });
      return ValueVector(outputValues);
    });

    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...super.parameters(),
      ...keyProj.parameters(),
      ...valueProj.parameters(),
    ];
  }
}
