import '../tensor/tensor.dart';
import 'layer.dart';
import 'module.dart';

class FeedForward extends Module {
  final Layer w1; // Expands dimension (usually 4x)
  final Layer w2; // Contracts back to embedSize

  FeedForward(int dim)
      : w1 = Layer(dim, dim * 4, useGelu: true),
        w2 = Layer(dim * 4, dim, useGelu: false);

  Tensor forward(Tensor x) => w2.forward(w1.forward(x));

  @override
  List<Tensor> parameters() => [...w1.parameters(), ...w2.parameters()];
}
