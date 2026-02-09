// file: transformer_encoder_block.dart

import '/nn/module.dart';
import 'multi_head_aft.dart'; // Import the AFT version
import 'feed_forward.dart';
import 'layer_norm2.dart'; 
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A single Transformer Encoder block using AFT.
///
/// Replaces standard dot-product attention with the Attention Free mechanism,
/// significantly reducing memory complexity for long sequences.
class TransformerEncoderBlock extends Module {
  final MultiHeadAFT attention; // Use AFT
  final FeedForward ffn;
  final LayerNorm ln1; 
  final LayerNorm ln2; 
  final int embedSize;

  // Added maxSeqLen to constructor to accommodate AFT position biases
  TransformerEncoderBlock(this.embedSize, int numHeads, int maxSeqLen)
      : attention = MultiHeadAFT(numHeads, embedSize, maxSeqLen,
            masked: false), // Encoder uses full context (unmasked)
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize);

  /// The forward pass flow remains identical to a standard Transformer,
  /// preserving the residual and LayerNorm (pre-norm) structure.
  List<ValueVector> forward(List<ValueVector> x) {
    final T = x.length;

    // 1. AFT Self-Attention sub-layer
    final x_norm1 = List.generate(T, (i) => ln1.forward(x[i]));
    final aft_out = attention.forward(x_norm1);
    final x_res1 = List.generate(T, (i) => x[i] + aft_out[i]);

    // 2. Feed-Forward sub-layer
    final x_norm2 = List.generate(T, (i) => ln2.forward(x_res1[i]));
    final ffn_out = List.generate(T, (i) => ffn.forward(x_norm2[i]));
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