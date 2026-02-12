import '../tensor/tensor.dart';
import 'feed_forward.dart';
import 'layer_norm.dart';
import 'module.dart';
import 'multi_head_attention.dart';

class TransformerBlock extends Module {
  final MultiHeadAttention attention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;

  TransformerBlock(int embedSize, int numHeads)
      : attention = MultiHeadAttention(embedSize, numHeads),
        ffn = FeedForward(embedSize), // Assume this is a 2-layer Tensor MLP
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  Tensor forward(Tensor x, {bool masked = false}) {
    // x shape: [seqLength, embedSize]

    // 1. Multi-Head Attention Sub-layer
    // x = x + Attention(LayerNorm(x))
    Tensor xNorm1 = ln1.forward(x);
    Tensor attnOut = attention.forward(xNorm1, masked: masked);
    Tensor xRes1 = x + attnOut; // Element-wise addition

    // 2. Feed-Forward Sub-layer
    // x = x + FFN(LayerNorm(x))
    Tensor xNorm2 = ln2.forward(xRes1);
    Tensor ffnOut = ffn.forward(xNorm2);
    Tensor out = xRes1 + ffnOut;

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
