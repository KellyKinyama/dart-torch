// In a new file: window_attention.dart

import '/nn/module.dart';
import '../transformer/aft_multi_head_attention.dart'; // Re-use MultiHeadAttention
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// Window-based Multi-Head Self-Attention.
///
/// This module wraps MultiHeadAttention to clarify its role in processing windows.
/// The actual window partitioning happens *before* calling this module.
class WindowAttention extends Module {
  final MultiHeadAFT attn;

  WindowAttention(int embedSize, int numHeads)
      : attn = MultiHeadAFT(numHeads, embedSize, 128
            // masked: false
            ); // Self-attention, not masked for decoder

  /// Forward pass for Window Attention.
  ///
  /// Takes a list of `ValueVector`s representing the tokens *within a single window*.
  List<ValueVector> forward(List<ValueVector> windowTokens) {
    // MultiHeadAttention handles the QKV projection, scaling, softmax, and output projection
    return attn.forward(windowTokens);
  }

  @override
  List<Value> parameters() {
    return attn.parameters();
  }
}
