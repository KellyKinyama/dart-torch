import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';
import 'self_attention.dart';

class MultiHeadAttention extends Module {
  final List<SelfAttention> heads;
  final Layer proj;
  final int numHeads;

  MultiHeadAttention(int embedSize, this.numHeads)
      : assert(embedSize % numHeads == 0),
        heads = List.generate(
            numHeads, (_) => SelfAttention(embedSize, embedSize ~/ numHeads)),
        proj = Layer(embedSize, embedSize, useGelu: false);

  Tensor forward(Tensor x, {bool masked = false}) {
    // x: [T, embedSize]

    // 1. Parallel Heads
    List<Tensor> headOuts =
        heads.map((h) => h.forward(x, masked: masked)).toList();

    // 2. Concatenate: [T, headSize] * numHeads -> [T, embedSize]
    Tensor combined = Tensor.concat(headOuts, axis: 1);

    // 3. Final Projection
    return proj.forward(combined);
  }

  @override
  List<Tensor> parameters() =>
      [...heads.expand((h) => h.parameters()), ...proj.parameters()];
}
