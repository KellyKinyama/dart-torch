// file: transformer_encoder_block.dart

import '/nn/module.dart';
import 'multi_head_attention.dart';
import '../transformer/feed_forward.dart';
import '../transformer/layer_norm2.dart'; // Using the more concise LayerNorm
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A single Transformer Encoder block using the Linformer attention mechanism.
///
/// This block consists of a Multi-Head Self-Attention layer and a Feed-Forward Network,
/// each with residual connections and layer normalization (pre-normalization).
class TransformerEncoderBlock extends Module {
  final MultiHeadAttention attention;
  final FeedForward ffn;
  final LayerNorm ln1; // LayerNorm before attention
  final LayerNorm ln2; // LayerNorm before FFN
  final int embedSize;
  final int? projK; // The projected sequence length for Linformer

  TransformerEncoderBlock(this.embedSize, int numHeads, {this.projK})
      : attention = MultiHeadAttention(numHeads, embedSize,
            masked: false, projK: projK), // Pass projK to the attention layer
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// The forward pass through a single Transformer Encoder block.
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length; // Sequence length

    // 1. First sub-layer: Multi-Head Self-Attention with pre-normalization and residual connection
    // Apply Layer Normalization to the input of the attention layer
    final x_norm1 = List.generate(T, (i) => ln1.forward(x[i]));
    // Apply attention
    final attn_out = attention.forward(x_norm1);
    // Add residual connection: original input + attention output
    final x_res1 = List.generate(T, (i) => x[i] + attn_out[i]);

    // 2. Second sub-layer: Feed-Forward Network with pre-normalization and residual connection
    // Apply Layer Normalization to the input of the FFN
    final x_norm2 = List.generate(T, (i) => ln2.forward(x_res1[i]));
    // Apply FFN
    final ffn_out = List.generate(T, (i) => ffn.forward(x_norm2[i]));
    // Add residual connection: input to FFN + FFN output
    final x_res2 = List.generate(T, (i) => x_res1[i] + ffn_out[i]);

    return x_res2;
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
