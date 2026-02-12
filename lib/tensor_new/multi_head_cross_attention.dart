import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class MultiHeadCrossAttention extends Module {
  final List<CrossAttentionHead> heads;
  final Layer proj;

  MultiHeadCrossAttention(int numHeads, int embedSize, int encoderEmbedSize)
      : assert(embedSize % numHeads == 0),
        heads = List.generate(
          numHeads,
          (_) => CrossAttentionHead(
              embedSize, encoderEmbedSize, embedSize ~/ numHeads),
        ),
        proj = Layer(embedSize, embedSize, useGelu: false);

  Tensor forward(Tensor x, Tensor encoderOut) {
    // x: Decoder sequence [T_dec, embedSize]
    // encoderOut: Encoder sequence [T_enc, encoderEmbedSize]

    List<Tensor> headOuts = heads.map((h) => h.forward(x, encoderOut)).toList();
    Tensor combined = Tensor.concat(headOuts, axis: 1);
    return proj.forward(combined);
  }

  @override
  List<Tensor> parameters() => [
        ...heads.expand((h) => h.parameters()),
        ...proj.parameters(),
      ];
}

class CrossAttentionHead extends Module {
  final Layer queryLayer;
  final Layer keyLayer;
  final Layer valueLayer;
  final int headSize;

  CrossAttentionHead(int embedSize, int encoderEmbedSize, this.headSize)
      : queryLayer = Layer(embedSize, headSize, useGelu: false),
        keyLayer = Layer(encoderEmbedSize, headSize, useGelu: false),
        valueLayer = Layer(encoderEmbedSize, headSize, useGelu: false);

  Tensor forward(Tensor x, Tensor encoderOut) {
    Tensor q = queryLayer.forward(x); // [T_dec, headSize]
    Tensor k = keyLayer.forward(encoderOut); // [T_enc, headSize]
    Tensor v = valueLayer.forward(encoderOut); // [T_enc, headSize]

    Tensor weights = q.matmul(k.transpose()) * (1.0 / math.sqrt(headSize));
    Tensor probs = weights.softmax(); // [T_dec, T_enc]
    return probs.matmul(v); // [T_dec, headSize]
  }

  @override
  List<Tensor> parameters() => [
        ...queryLayer.parameters(),
        ...keyLayer.parameters(),
        ...valueLayer.parameters()
      ];
}
