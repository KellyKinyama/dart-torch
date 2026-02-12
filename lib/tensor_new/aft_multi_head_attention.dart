import '../tensor/tensor.dart';
import 'aft_attention.dart';
import 'layer.dart';
import 'module.dart';

class MultiHeadAFT extends Module {
  final List<AFTAttention> heads;
  final Layer proj;
  final int numHeads;
  final int headSize;
  final bool masked; // Store the masking state

  MultiHeadAFT(this.numHeads, int embedSize, int maxSeqLen, {this.masked = false})
      : assert(embedSize % numHeads == 0),
        headSize = embedSize ~/ numHeads,
        heads = List.generate(
            numHeads,
            (i) => AFTAttention(embedSize, embedSize ~/ numHeads, maxSeqLen, 
                masked: masked)), // Pass 'masked' down to the AFTAttention heads
        proj = Layer(embedSize, embedSize, useGelu: false);

  Tensor forward(Tensor x) {
    // Each head returns [T, headSize]
    final List<Tensor> headOutputs = heads.map((h) => h.forward(x)).toList();

    // Concatenate to [T, embedSize]
    final Tensor concatenated = Tensor.concat(headOutputs, axis: 1);

    // Final linear projection
    return proj.forward(concatenated);
  }

  @override
  List<Tensor> parameters() => [
        ...heads.expand((h) => h.parameters()),
        ...proj.parameters(),
      ];
}