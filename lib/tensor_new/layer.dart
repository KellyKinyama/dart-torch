import '../tensor/tensor.dart';
import 'module.dart';

// class Layer extends Module {
//   final Tensor weights;
//   final Tensor bias;
//   final bool useRelu;

//   Layer(int nin, int nout, {this.useRelu = true})
//       : weights = Tensor.xavier([nin, nout]),
//         bias = Tensor.fill([1, nout], 0.0);

//   Tensor forward(Tensor x) {
//     // x is [batch_size, nin], weights is [nin, nout]
//     // out becomes [batch_size, nout]
//     Tensor act = x.matmul(weights) + bias;
//     return useRelu ? act.relu() : act;
//   }

//   @override
//   List<Tensor> parameters() => [weights, bias];
// }

class Layer extends Module {
  final Tensor weights;
  final Tensor bias;
  final bool useGelu;

  Layer(int nIn, int nOut, {this.useGelu = true})
      : weights = Tensor.xavier([nIn, nOut]),
        bias = Tensor.zeros([1, nOut]);

  Tensor forward(Tensor x) {
    // Standard Linear Transformation: xW + b
    Tensor out = x.matmul(weights) + bias;

    // Use GELU instead of ReLU to prevent "Dead Neurons"
    return useGelu ? out.gelu() : out;
  }

  @override
  List<Tensor> parameters() => [weights, bias];
}
