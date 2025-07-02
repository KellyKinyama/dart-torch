// file: transformer_block.dart

import '/nn/module.dart';
import 'multi_head_attention.dart';
import 'feed_forward.dart';
import 'layer_norm2.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A single Transformer block, which combines attention and feed-forward layers.
class TransformerBlock extends Module {
  final MultiHeadAttention attention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final int embedSize;

  TransformerBlock(this.embedSize, int numHeads, {bool masked = false})
      : attention = MultiHeadAttention(numHeads, embedSize, masked: masked),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// The forward pass through a single Transformer block.
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length;

    // 1. First sub-layer: Multi-Head Attention with pre-normalization and residual connection
    final x_norm1 = List.generate(T, (i) => ln1.forward(x[i]));
    final attn_out = attention.forward(x_norm1);
    final x_res1 = List.generate(T, (i) => x[i] + attn_out[i]);

    // 2. Second sub-layer: Feed-Forward with pre-normalization and residual connection
    final x_norm2 = List.generate(T, (i) => ln2.forward(x_res1[i]));
    final ffn_out = List.generate(T, (i) => ffn.forward(x_norm2[i]));
    final out = List.generate(T, (i) => x_res1[i] + ffn_out[i]);

    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...attention.parameters(),
      ...ffn.parameters(),
      ...ln1.parameters(),
      ...ln2.parameters(),
    ];
  }
}
