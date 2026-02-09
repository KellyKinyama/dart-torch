// file: multi_head_attention.dart

import '/nn/module.dart';
import 'aft_attention.dart'; // Import the new AFT class
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';

/// Combines multiple AFT heads to run in parallel.
class MultiHeadAFT extends Module {
  final List<AFTAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;

  MultiHeadAFT(this.numHeads, int embedSize, int maxSeqLen)
      : assert(embedSize % numHeads == 0),
        headSize = embedSize ~/ numHeads,
        // Replace SelfAttention with AFTAttention
        heads = List.generate(numHeads,
            (i) => AFTAttention(embedSize, embedSize ~/ numHeads, maxSeqLen)),
        proj = Layer.fromNeurons(embedSize, embedSize);

  /// Forward pass remains structurally identical, but uses AFT logic internally.
  List<ValueVector> forward(List<ValueVector> x) {
    // 1. Compute AFT head outputs
    // Each head performs the weighted exp-bias aggregation
    final headOutputs = heads.map((h) => h.forward(x)).toList();

    // 2. Concatenate head outputs
    final T = x.length;
    final concatenated = List.generate(T, (i) {
      final values =
          headOutputs.expand((head_out) => head_out[i].values).toList();
      return ValueVector(values);
    });

    // 3. Apply final projection
    return concatenated.map((c) => proj.forward(c)).toList();
  }

  @override
  List<Value> parameters() {
    return [
      ...heads.expand((h) => h.parameters()),
      ...proj.parameters(),
    ];
  }
}
