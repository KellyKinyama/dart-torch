import '/nn/module.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';
import 'aft_self_attention.dart';
// import 'aft_attention.dart'; // This is the AFT version of SelfAttention

class MultiHeadAFT extends Module {
  final List<AFTAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;

  MultiHeadAFT(this.numHeads, int embedSize, int maxSeqLen,
      {bool masked = false})
      : assert(embedSize % numHeads == 0),
        headSize = embedSize ~/ numHeads,
        heads = List.generate(
            numHeads,
            (i) => AFTAttention(embedSize, embedSize ~/ numHeads, maxSeqLen,
                masked: masked)),
        proj = Layer.fromNeurons(embedSize, embedSize);

  List<ValueVector> forward(List<ValueVector> x) {
    final headOutputs = heads.map((h) => h.forward(x)).toList();
    final T = x.length;

    // Concatenate results from all AFT heads
    final concatenated = List.generate(T, (i) {
      final values = headOutputs.expand((h_out) => h_out[i].values).toList();
      return ValueVector(values);
    });

    return concatenated.map((c) => proj.forward(c)).toList();
  }

  @override
  List<Value> parameters() => [
        ...heads.expand((h) => h.parameters()),
        ...proj.parameters(),
      ];
}
