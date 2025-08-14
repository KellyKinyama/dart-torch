import 'value.dart';
import 'value_vector.dart';

abstract class Module {
  void zeroGrad() {
    for (var p in parameters()) {
      p.grad = 0;
    }
  }

  // List<ValueVector> forward(List<ValueVector> x);

  List<Value> parameters() {
    throw UnimplementedError("unimplemented");
  }
}
