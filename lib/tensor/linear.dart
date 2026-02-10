import 'module.dart';
import 'tensor.dart';

class Linear extends Module {
  final Tensor weights;
  final Tensor bias;

  Linear(int nin, int nout)
      : weights = Tensor.xavier([nin, nout]), // Use Xavier here!
        bias = Tensor.fill([1, nout], 0.0);

  Tensor forward(Tensor x) {
    // Standard Y = XW + B
    final out = x.matmul(weights) + bias;

    // We don't need to define a new onBackward here because
    // matmul and + already have their own onBackward logic
    // defined inside the Tensor class.
    return out;
  }

  @override
  List<Tensor> parameters() => [weights, bias];
}
