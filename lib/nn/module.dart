import 'value.dart';

class Module {
  void zeroGrad() {
    for (var p in parameters()) {
      p.grad = 0;
    }
  }

  List<Value> parameters() {
    throw UnimplementedError("unimplemented");
  }
}
