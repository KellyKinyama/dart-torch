// file: self_attention.dart

import '../transformer/self_attention.dart';
// import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';
import 'dart:math' as math;
// import 'package:math/math.dart';

/// A single self-attention head.
// class SelfAttention extends Module {
//   final Layer key;
//   final Layer query;
//   final Layer value;
//   final bool masked;
//   final int headSize;

//   SelfAttention(int embedSize, this.headSize, {this.masked = false})
//       : key = Layer.fromNeurons(embedSize, headSize),
//         query = Layer.fromNeurons(embedSize, headSize),
//         value = Layer.fromNeurons(embedSize, headSize);

//   /// Forward pass for a single self-attention head.
//   List<ValueVector> forward(List<ValueVector> x) {
//     final T = x.length;
//     final k = x.map((v) => key.forward(v)).toList();
//     final q = x.map((v) => query.forward(v)).toList();
//     final v = x.map((v) => value.forward(v)).toList();

//     final List<List<Value>> attention = List.generate(T, (i) {
//       return List.generate(T, (j) {
//         final score = q[i].dot(k[j]);
//         if (masked && i < j) {
//           return Value(double.negativeInfinity);
//         }

//         // return ValueVector(row) * Value(1.0 / math.sqrt(headSize.toDouble()));
//         return score / math.sqrt(headSize);
//       });
//     });

//     final weights = attention.map((row) => ValueVector(row).softmax()).toList();

//     final List<ValueVector> out = List.generate(T, (i) {
//       final List<Value> outputValues = List.generate(headSize, (j) {
//         Value sum = Value(0.0);
//         for (int t = 0; t < T; t++) {
//           sum += weights[i].values[t] * v[t].values[j];
//         }
//         return sum;
//       });
//       return ValueVector(outputValues);
//     });

//     return out;
//   }

//   @override
//   List<Value> parameters() {
//     return [
//       ...key.parameters(),
//       ...query.parameters(),
//       ...value.parameters(),
//     ];
//   }
// }

// NEW: Linformer variant of the self-attention head
class LinformerSelfAttention extends SelfAttention {
  final Layer keyProj;
  final Layer valueProj;
  final int projK; // The projected sequence length

  LinformerSelfAttention(int embedSize, int headSize,
      {bool masked = false, required this.projK})
      : keyProj = Layer.fromNeurons(embedSize, projK),
        valueProj = Layer.fromNeurons(embedSize, projK),
        super(embedSize, headSize, masked: masked);

  @override
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length;
    // 1. Linearly project the keys and values to a smaller sequence length (k)
    final projectedKeys = x.map((v) => keyProj.forward(v)).toList();
    final projectedValues = x.map((v) => valueProj.forward(v)).toList();
    final q = x.map((v) => query.forward(v)).toList();

    // 2. Calculate attention with the projected key and value
    final List<List<Value>> attention = List.generate(T, (i) {
      return List.generate(T, (j) {
        final score = q[i].dot(projectedKeys[j]);
        if (masked && i < j) {
          return Value(double.negativeInfinity);
        }
        return score / math.sqrt(headSize);
      });
    });

    final weights = attention.map((row) => ValueVector(row).softmax()).toList();

    final List<ValueVector> out = List.generate(T, (i) {
      final List<Value> outputValues = List.generate(headSize, (j) {
        Value sum = Value(0.0);
        for (int t = 0; t < T; t++) {
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
