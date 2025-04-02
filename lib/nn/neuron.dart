import 'dart:math' as math;

import 'module.dart';
import 'value.dart';
import 'value_vector.dart';

class Neuron extends Module {
  ValueVector w;
  Value? b;
  bool nonlin = true;

  Neuron(this.w, {this.b, this.nonlin = true}) : super();

  factory Neuron.fromWeights(int nin, {bool nonlin = true}) {
    final w = List<Value>.generate(
        nin, (int index) => Value(math.Random().nextDouble()),
        growable: false);

    /// print(fixedLengthList); // [0, 1, 4]

    Value b = Value(0.5);

    return Neuron(ValueVector(w), b: b, nonlin: true);
  }

  Value forward(ValueVector x) {
    final matMul = w.dot(x);

    return b == null ? matMul : matMul + b;
  }

  @override
  List<Value> parameters() {
    // TODO: implement parameters
    return [...w.values, b!];
  }
}
