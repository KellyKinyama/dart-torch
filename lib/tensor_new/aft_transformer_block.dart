import '../tensor/tensor.dart';
import 'aft_multi_head_attention.dart';
import 'feed_forward.dart';
import 'layer_norm.dart';
import 'module.dart';

/// A single Transformer block using AFT (Tensor-based).
/// Implements the Pre-Norm architecture for better training stability.
class TransformerBlock extends Module {
  final MultiHeadAFT attention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final int embedSize;

  TransformerBlock(this.embedSize, int numHeads, int maxSeqLen,
      {bool masked = false})
      : attention =
            MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: masked),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// x: Input Tensor of shape [T, embedSize]
  Tensor forward(Tensor x) {
    // 1. Attention Sub-layer (Pre-Norm)
    // ln1(x) -> MultiHeadAFT -> + x (Residual)
    Tensor x_norm1 = ln1.forward(x);
    Tensor aft_out = attention.forward(x_norm1);
    Tensor x_res1 = x + aft_out;

    // 2. Feed-Forward Sub-layer (Pre-Norm)
    // ln2(x_res1) -> FFN -> + x_res1 (Residual)
    Tensor x_norm2 = ln2.forward(x_res1);
    Tensor ffn_out = ffn.forward(x_norm2);
    Tensor out = x_res1 + ffn_out;

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...attention.parameters(),
        ...ffn.parameters(),
        ...ln1.parameters(),
        ...ln2.parameters(),
      ];
}
