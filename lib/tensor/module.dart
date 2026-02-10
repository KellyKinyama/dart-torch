import 'tensor.dart';

abstract class Module {
  /// Returns all trainable tensors (weights, biases, etc.)
  List<Tensor> parameters();

  void zeroGrad() {
    for (var p in parameters()) {
      p.zeroGrad();
    }
  }
}
