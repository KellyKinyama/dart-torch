// file: multi_head_attention.dart

// import '../transformer/cross_attention.dart';
import '/nn/module.dart';
import 'self_attention.dart';
import '../transformer/self_attention.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// Combines multiple self-attention heads to run in parallel.
class MultiHeadAttention extends Module {
  final List<SelfAttention> heads;
  final Layer proj; // Final linear projection layer
  final int numHeads;
  final int headSize;

  MultiHeadAttention(this.numHeads, int embedSize,
      {bool masked = false, int? projK})
      : assert(embedSize % numHeads == 0),
        headSize = embedSize ~/ numHeads,
        heads = projK != null
            ? List.generate(
                numHeads,
                (i) => LinformerSelfAttention(embedSize, embedSize ~/ numHeads,
                    masked: masked, projK: projK))
            : List.generate(
                numHeads,
                (i) => SelfAttention(embedSize, embedSize ~/ numHeads,
                    masked: masked)),
        proj = Layer.fromNeurons(embedSize, embedSize);

  /// Forward pass for Multi-Head Attention.
  List<ValueVector> forward(List<ValueVector> x) {
    // 1. Compute all head outputs in parallel
    final headOutputs = heads.map((h) => h.forward(x)).toList();

    // 2. Concatenate head outputs for each token position
    final T = x.length;
    final concatenated = List.generate(T, (i) {
      final values =
          headOutputs.expand((head_out) => head_out[i].values).toList();
      return ValueVector(values);
    });

    // 3. Apply final projection layer
    final out = concatenated.map((c) => proj.forward(c)).toList();
    return out;
  }

  @override
  List<Value> parameters() {
    return [
      ...heads.expand((h) => h.parameters()),
      ...proj.parameters(),
    ];
  }
}
