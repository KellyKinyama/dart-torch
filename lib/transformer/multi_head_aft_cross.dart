import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';
import 'aft_cross_attention.dart'; // The implementation we discussed previously

class MultiHeadAFTCross extends Module {
  final List<AFTCrossAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;

  MultiHeadAFTCross(this.numHeads, int decoderEmbedSize, int encoderEmbedSize,
      int maxTDec, int maxTEnc)
      : assert(decoderEmbedSize % numHeads == 0),
        headSize = decoderEmbedSize ~/ numHeads,
        heads = List.generate(
            numHeads,
            (i) => AFTCrossAttention(decoderEmbedSize, encoderEmbedSize,
                decoderEmbedSize ~/ numHeads, maxTDec, maxTEnc)),
        proj = Layer.fromNeurons(decoderEmbedSize, decoderEmbedSize);

  List<ValueVector> forward(List<ValueVector> xDec, List<ValueVector> xEnc) {
    final headOutputs = heads.map((h) => h.forward(xDec, xEnc)).toList();
    final T_dec = xDec.length;

    final concatenated = List.generate(T_dec, (i) {
      final values = headOutputs.expand((h_out) => h_out[i].values).toList();
      return ValueVector(values);
    });

    return concatenated.map((c) => proj.forward(c)).toList();
  }

  @override
  List<Value> parameters() => [
        ...heads.expand((h) => h.parameters()),
        ...proj.parameters(),
      ];
}
