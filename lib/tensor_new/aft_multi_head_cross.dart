import '../tensor/tensor.dart';
import 'aft_cross_attention.dart';
import 'layer.dart';
import 'module.dart';

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
        // Final projection to merge head outputs back to the decoder's embedding space
        proj = Layer(decoderEmbedSize, decoderEmbedSize, useGelu: false);

  /// xDec: [T_dec, decoderEmbedSize]
  /// xEnc: [T_enc, encoderEmbedSize]
  Tensor forward(Tensor xDec, Tensor xEnc) {
    // 1. Run each Cross-Attention head
    // Each head produces a Tensor of shape [T_dec, headSize]
    final headOutputs = heads.map((h) => h.forward(xDec, xEnc)).toList();

    // 2. Concatenate heads along the feature dimension (axis 1)
    // Resulting shape: [T_dec, decoderEmbedSize]
    final concatenated = Tensor.concat(headOutputs, axis: 1);

    // 3. Final linear projection
    return proj.forward(concatenated);
  }

  @override
  List<Tensor> parameters() => [
        ...heads.expand((h) => h.parameters()),
        ...proj.parameters(),
      ];
}
