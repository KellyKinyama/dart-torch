import 'aft_multi_head_attention.dart';
import 'aft_multi_head_cross.dart';
import 'layer_norm.dart';
import 'mlp.dart';
import 'module.dart';
import 'tensor.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final MLP ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;

  TransformerDecoderBlock(
      int embedSize, int numHeads, int encoderEmbedSize, int maxSeqLen)
      : selfAttention =
            MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: true),
        crossAttention =
            MultiHeadAFTCross(numHeads, embedSize, encoderEmbedSize, maxSeqLen),
        ffn = MLP(embedSize, [4 * embedSize, embedSize]),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize),
        ln3 = LayerNorm(embedSize);

  /// x_dec: [T_dec, EmbedSize]
  /// x_enc: [T_enc, EncoderEmbedSize]
  Tensor forward(Tensor x_dec, Tensor x_enc) {
    // 1. Masked Self-Attention + Residual
    final x1 = x_dec + selfAttention.forward(ln1.forward(x_dec));

    // 2. Cross-Attention + Residual (Query from decoder, Key/Value from encoder)
    final x2 = x1 + crossAttention.forward(ln2.forward(x1), x_enc);

    // 3. Feed-Forward + Residual
    final out = x2 + ffn.forward(ln3.forward(x2));

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...selfAttention.parameters(),
        ...crossAttention.parameters(),
        ...ffn.parameters(),
        ...ln1.parameters(),
        ...ln2.parameters(),
        ...ln3.parameters(),
      ];
}
