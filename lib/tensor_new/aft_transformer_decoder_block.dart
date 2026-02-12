import '../tensor/tensor.dart';
import 'aft_multi_head_attention.dart';
import 'aft_multi_head_cross.dart';
import 'feed_forward.dart';
import 'layer_norm.dart';
import 'module.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;
  final int embedSize;

  TransformerDecoderBlock(
      this.embedSize, int numHeads, int encoderEmbedSize, int maxSeqLen)
      : selfAttention =
            MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: true),
        crossAttention = MultiHeadAFTCross(
            numHeads, embedSize, encoderEmbedSize, maxSeqLen, maxSeqLen),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize),
        ln3 = LayerNorm(embedSize);

  /// x_decoder: [T_dec, embedSize]
  /// x_encoder: [T_enc, encoderEmbedSize]
  Tensor forward(Tensor x_decoder, Tensor x_encoder) {
    // 1. Masked AFT Self-Attention + Residual
    // We normalize the whole matrix [T, D] at once
    Tensor x_norm1 = ln1.forward(x_decoder);
    Tensor self_attn_out = selfAttention.forward(x_norm1);
    Tensor x_res1 = x_decoder + self_attn_out;

    // 2. AFT Cross-Attention + Residual
    Tensor x_norm2 = ln2.forward(x_res1);
    Tensor cross_attn_out = crossAttention.forward(x_norm2, x_encoder);
    Tensor x_res2 = x_res1 + cross_attn_out;

    // 3. Feed-Forward + Residual
    Tensor x_norm3 = ln3.forward(x_res2);
    Tensor ffn_out = ffn.forward(x_norm3);
    Tensor out = x_res2 + ffn_out;

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
