// file: transformer_decoder_block.dart

import '/nn/module.dart';
import 'multi_head_aft.dart'; // Masked AFT for self-attention
import 'multi_head_aft_cross.dart'; // NEW: Multi-Head version of AFT-Cross
// import 'aft_multi_head_attention.dart';
import 'feed_forward.dart';
import 'layer_norm2.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention; // Attention Free
  final MultiHeadAFTCross crossAttention; // Attention Free
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;
  final int embedSize;

  TransformerDecoderBlock(this.embedSize, int numHeads, int encoderEmbedSize,
      int maxSeqLen // Required for AFT learned biases
      )
      : selfAttention =
            MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: true),
        crossAttention = MultiHeadAFTCross(
            numHeads, embedSize, encoderEmbedSize, maxSeqLen, maxSeqLen),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize),
        ln3 = LayerNorm(embedSize);

  List<ValueVector> forward(
      List<ValueVector> x_decoder, List<ValueVector> x_encoder) {
    final T_dec = x_decoder.length;

    // 1. Masked AFT Self-Attention
    final x_norm1 = List.generate(T_dec, (i) => ln1.forward(x_decoder[i]));
    final self_attn_out = selfAttention.forward(x_norm1);
    final x_res1 = List.generate(T_dec, (i) => x_decoder[i] + self_attn_out[i]);

    // 2. AFT Cross-Attention
    final x_norm2 = List.generate(T_dec, (i) => ln2.forward(x_res1[i]));
    final cross_attn_out = crossAttention.forward(x_norm2, x_encoder);
    final x_res2 = List.generate(T_dec, (i) => x_res1[i] + cross_attn_out[i]);

    // 3. Feed-Forward
    final x_norm3 = List.generate(T_dec, (i) => ln3.forward(x_res2[i]));
    final ffn_out = List.generate(T_dec, (i) => ffn.forward(x_norm3[i]));
    final out = List.generate(T_dec, (i) => x_res2[i] + ffn_out[i]);

    return out;
  }

  @override
  List<Value> parameters() => [
        ...selfAttention.parameters(),
        ...crossAttention.parameters(),
        ...ffn.parameters(),
        ...ln1.parameters(),
        ...ln2.parameters(),
        ...ln3.parameters(),
      ];
}
