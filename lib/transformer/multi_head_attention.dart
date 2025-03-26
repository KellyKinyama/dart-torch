import 'dart:math';

import '../nn/value.dart';
import '../nn/value_vector.dart';
import 'linear_layer.dart';

// class MultiHeadAttention {
//   final int numHeads;
//   final int embedDim;
//   final int headDim;
//   final List<LinearLayer> qLayers;
//   final List<LinearLayer> kLayers;
//   final List<LinearLayer> vLayers;
//   final LinearLayer outputLayer;

//   MultiHeadAttention(this.embedDim, this.numHeads)
//       : headDim = embedDim ~/ numHeads,
//         qLayers = [],
//         kLayers = [],
//         vLayers = [],
//         outputLayer = LinearLayer(embedDim, embedDim) {
//     for (var _ = 0; _ < numHeads; _++) {
//       qLayers.add(LinearLayer(embedDim, headDim));
//       kLayers.add(LinearLayer(embedDim, headDim));
//       vLayers.add(LinearLayer(embedDim, headDim));
//     }
//   }

//   List<ValueVector> forward(List<ValueVector> x) {
//     final List<ValueVector> allHeads = [];

//     for (var i = 0; i < numHeads; i++) {
//       final qProj = x.map((v) => qLayers[i].forward(v)).toList();
//       final kProj = x.map((v) => kLayers[i].forward(v)).toList();
//       final vProj = x.map((v) => vLayers[i].forward(v)).toList();

//       final List<List<Value>> scores = [];
//       for (var q in qProj) {
//         final row = kProj
//             .map((k) => q.dot(k) / Value(sqrt(headDim.toDouble())))
//             .toList();
//         scores.add(row);
//       }

//       final List<List<Value>> attnWeights = scores.map((row) {
//         final expRow = row.map((v) => v.exp()).toList();
//         final sumExp = expRow.fold(Value(0.0), (a, b) => a + b);
//         return expRow.map((v) => v / sumExp).toList();
//       }).toList();

//       final List<ValueVector> attnOutput = [];
//       for (var i = 0; i < attnWeights.length; i++) {
//         final weightedSum = List<Value>.filled(headDim, Value(0.0));
//         for (var j = 0; j < attnWeights[i].length; j++) {
//           final vj = vProj[j];
//           for (var k = 0; k < headDim; k++) {
//             weightedSum[k] = weightedSum[k] + vj.values[k] * attnWeights[i][j];
//           }
//         }
//         attnOutput.add(ValueVector(weightedSum));
//       }

//       allHeads.addAll(attnOutput);
//     }

//     final List<ValueVector> combined = [];
//     for (var i = 0; i < x.length; i++) {
//       final concat = <Value>[];
//       for (var j = 0; j < numHeads; j++) {
//         concat.addAll(allHeads[i + j * x.length].values);
//       }
//       combined.add(outputLayer.forward(ValueVector(concat)));
//     }

//     return combined;
//   }

//   @override
//   String toString() =>
//       'MultiHeadAttention(embedDim=$embedDim, heads=$numHeads)';
// }

class MultiHeadAttention {
  final int numHeads;
  final int embedDim;
  final int headDim;
  final List<LinearLayer> qLayers;
  final List<LinearLayer> kLayers;
  final List<LinearLayer> vLayers;
  final LinearLayer outputLayer;

  MultiHeadAttention(this.embedDim, this.numHeads)
      : headDim = embedDim ~/ numHeads,
        qLayers = [],
        kLayers = [],
        vLayers = [],
        outputLayer = LinearLayer(embedDim, embedDim) {
    if (headDim == 0) {
      throw ArgumentError(
          "embedDim must be divisible by numHeads. Received embedDim=$embedDim, numHeads=$numHeads.");
    }

    for (var _ = 0; _ < numHeads; _++) {
      qLayers.add(LinearLayer(embedDim, headDim));
      kLayers.add(LinearLayer(embedDim, headDim));
      vLayers.add(LinearLayer(embedDim, headDim));
    }
  }

  List<ValueVector> forward(List<ValueVector> x) {
    final List<ValueVector> allHeads = [];

    for (var i = 0; i < numHeads; i++) {
      final qProj = x.map((v) => qLayers[i].forward(v)).toList();
      final kProj = x.map((v) => kLayers[i].forward(v)).toList();
      final vProj = x.map((v) => vLayers[i].forward(v)).toList();

      final List<List<Value>> scores = [];
      for (var q in qProj) {
        final row = kProj
            .map((k) => q.dot(k) / Value(sqrt(headDim.toDouble())))
            .toList();
        scores.add(row);
      }

      final List<List<Value>> attnWeights = scores.map((row) {
        final expRow = row.map((v) => v.exp()).toList();
        final sumExp = expRow.fold(Value(0.0), (a, b) => a + b);
        return expRow.map((v) => v / sumExp).toList();
      }).toList();

      final List<ValueVector> attnOutput = [];
      for (var i = 0; i < attnWeights.length; i++) {
        final weightedSum = List<Value>.filled(headDim, Value(0.0));
        for (var j = 0; j < attnWeights[i].length; j++) {
          final vj = vProj[j];
          for (var k = 0; k < headDim; k++) {
            weightedSum[k] = weightedSum[k] + vj.values[k] * attnWeights[i][j];
          }
        }
        attnOutput.add(ValueVector(weightedSum));
      }

      allHeads.addAll(attnOutput);
    }

    final List<ValueVector> combined = [];
    for (var i = 0; i < x.length; i++) {
      final concat = <Value>[];
      for (var j = 0; j < numHeads; j++) {
        concat.addAll(allHeads[i + j * x.length].values);
      }
      combined.add(outputLayer.forward(ValueVector(concat)));
    }

    return combined;
  }

  @override
  String toString() =>
      'MultiHeadAttention(embedDim=$embedDim, heads=$numHeads)';
}
