// file: transformer_block.dart

import '/nn/module.dart';
// import 'multi_head_aft.dart'; // Point to your AFT implementation file
import 'aft_multi_head_attention.dart';
import 'feed_forward.dart';
import 'layer_norm2.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A single Transformer block using Attention Free Transformer (AFT) logic.
class TransformerBlock extends Module {
  // 1. Change type from MultiHeadAttention to MultiHeadAFT
  final MultiHeadAFT attention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final int embedSize;

  // 2. Update constructor to include maxSeqLen
  TransformerBlock(this.embedSize, int numHeads, int maxSeqLen,
      {bool masked = false})
      : attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// The forward pass logic remains the same (Residual + LayerNorm),
  /// but the 'attention.forward' now executes the AFT linear-complexity logic.
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length;

    // 1. First sub-layer: AFT with pre-normalization and residual connection
    final x_norm1 = List.generate(T, (i) => ln1.forward(x[i]));
    final aft_out = attention.forward(x_norm1);
    final x_res1 = List.generate(T, (i) => x[i] + aft_out[i]);

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
