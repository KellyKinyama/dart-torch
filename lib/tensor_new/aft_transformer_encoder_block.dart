import '../tensor/tensor.dart';
import 'aft_multi_head_attention.dart';
import 'feed_forward.dart';
import 'layer_norm.dart';
import 'module.dart';

/// A single Transformer Encoder block using AFT (Tensor-based).
class TransformerEncoderBlock extends Module {
  final MultiHeadAFT attention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final int embedSize;

  TransformerEncoderBlock(this.embedSize, int numHeads, int maxSeqLen)
      : attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: false),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// x: [T, embedSize]
  Tensor forward(Tensor x) {
    // 1. AFT Self-Attention sub-layer (Pre-norm structure)
    // We normalize the entire block [T, D] at once.
    Tensor x_norm1 = ln1.forward(x);
    Tensor aft_out = attention.forward(x_norm1);
    Tensor x_res1 = x + aft_out; // Residual connection

    // 2. Feed-Forward sub-layer
    Tensor x_norm2 = ln2.forward(x_res1);
    Tensor ffn_out = ffn.forward(x_norm2);
    Tensor out = x_res1 + ffn_out; // Residual connection

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
