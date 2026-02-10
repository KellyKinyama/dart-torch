import 'aft_multi_head_attention.dart';
import 'module.dart';
import 'tensor.dart';
import 'mlp.dart'; // Your vectorized MLP/FeedForward class
import 'layer_norm.dart';

class TransformerEncoderBlock extends Module {
  final MultiHeadAFT attention;
  final MLP ffn; // Using the vectorized MLP class
  final LayerNorm ln1;
  final LayerNorm ln2;
  final int embedSize;

  TransformerEncoderBlock(this.embedSize, int numHeads, int maxSeqLen)
      : attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: false),
        // Standard FFN: expand to 4x, then contract back to embedSize
        ffn = MLP(embedSize, [4 * embedSize, embedSize]),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// Forward pass: x is [T, EmbedSize]
  Tensor forward(Tensor x) {
    // 1. AFT Self-Attention sub-layer (Pre-Norm + Residual)
    // x = x + Attention(LN1(x))
    final Tensor x_norm1 = ln1.forward(x);
    final Tensor aft_out = attention.forward(x_norm1);
    final Tensor x_res1 = x + aft_out;

    // 2. Feed-Forward sub-layer (Pre-Norm + Residual)
    // out = x_res1 + FFN(LN2(x_res1))
    final Tensor x_norm2 = ln2.forward(x_res1);
    final Tensor ffn_out = ffn.forward(x_norm2);
    final Tensor out = x_res1 + ffn_out;

    return out;
  }

  @override
  List<Tensor> parameters() {
    return [
      ...attention.parameters(),
      ...ffn.parameters(),
      ...ln1.parameters(),
      ...ln2.parameters(),
    ];
  }
}
