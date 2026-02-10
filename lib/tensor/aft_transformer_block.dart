import 'aft_multi_head_attention.dart';
import 'layer_norm.dart';
import 'mlp.dart';
import 'module.dart';
import 'tensor.dart';

class TransformerBlock extends Module {
  final MultiHeadAFT attention;
  final MLP ffn; // Using the MLP class we built earlier
  final LayerNorm ln1;
  final LayerNorm ln2;

  TransformerBlock(int embedSize, int numHeads, int maxSeqLen)
      : attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen),
        // FFN typically expands to 4x the dimension then back down
        ffn = MLP(embedSize, [4 * embedSize, embedSize]),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  @override
  Tensor forward(Tensor x) {
    // x shape: [T, EmbedSize]

    // 1. First sub-layer: Pre-Norm + AFT Attention + Residual
    // x = x + Attention(LN1(x))
    final attnOut = attention.forward(ln1.forward(x));
    final x1 = x + attnOut;

    // 2. Second sub-layer: Pre-Norm + Feed-Forward + Residual
    // out = x1 + FFN(LN2(x1))
    final ffnOut = ffn.forward(ln2.forward(x1));
    final out = x1 + ffnOut;

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
