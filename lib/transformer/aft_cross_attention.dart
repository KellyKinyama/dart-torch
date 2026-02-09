// file: aft_cross_attention.dart

import 'dart:math' as math;
import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// AFT adaptation for Cross-Attention.
class AFTCrossAttention extends Module {
  final Layer key; // Projects Encoder outputs
  final Layer query; // Projects Decoder inputs
  final Layer value; // Projects Encoder outputs
  final int headSize;

  // Biases for Decoder-Encoder interaction (T_decoder x T_encoder)
  final List<ValueVector> w;

  AFTCrossAttention(int decoderEmbedSize, int encoderEmbedSize, this.headSize,
      int maxTDec, int maxTEnc)
      : key = Layer.fromNeurons(encoderEmbedSize, headSize),
        query = Layer.fromNeurons(decoderEmbedSize, headSize),
        value = Layer.fromNeurons(encoderEmbedSize, headSize),
        w = List.generate(
            maxTDec,
            (_) => ValueVector(List.generate(maxTEnc,
                (_) => Value(math.Random().nextDouble() * 0.02 - 0.01))));

  List<ValueVector> forward(List<ValueVector> xDec, List<ValueVector> xEnc) {
    final TDec = xDec.length;
    final TEnc = xEnc.length;

    final q = xDec.map((v) => query.forward(v)).toList();
    final k = xEnc.map((v) => key.forward(v)).toList();
    final v = xEnc.map((v) => value.forward(v)).toList();

    final sigmoidQ = q.map((vec) => vec.sigmoid()).toList();

    return List.generate(TDec, (t) {
      ValueVector numerator = ValueVector(List.filled(headSize, Value(0.0)));
      ValueVector denominator = ValueVector(List.filled(headSize, Value(0.0)));

      // Aggregate context from all Encoder positions (t')
      for (int tp = 0; tp < TEnc; tp++) {
        final bias = w[t].values[tp];
        final expKW =
            ValueVector(k[tp].values.map((kv) => (kv + bias).exp()).toList());

        numerator += (expKW * v[tp]);
        denominator += expKW;
      }

      return sigmoidQ[t] * (numerator / denominator);
    });
  }

  @override
  List<Value> parameters() => [
        ...key.parameters(),
        ...query.parameters(),
        ...value.parameters(),
        ...w.expand((row) => row.values)
      ];
}
