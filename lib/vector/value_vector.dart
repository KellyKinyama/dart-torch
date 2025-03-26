import 'dart:math' as math;
import '../auto_diff2.dart';

class ValueVector {
  final List<Value> values;

  ValueVector(this.values);

  // Dot product with another ValueVector
  Value dot(ValueVector other) {
    assert(values.length == other.values.length);
    Value result = Value(0.0);
    for (int i = 0; i < values.length; i++) {
      result += values[i] * other.values[i];
    }
    return result;
  }

  // Add a Value scalar to each element
  ValueVector operator +(Value other) =>
      ValueVector(values.map((v) => v + other).toList());

  // Subtract another ValueVector
  ValueVector operator -(ValueVector other) {
    assert(values.length == other.values.length);
    return ValueVector(
        List.generate(values.length, (i) => values[i] - other.values[i]));
  }

  // Element-wise square
  ValueVector squared() => ValueVector(values.map((v) => v * v).toList());

  // Mean of all values
  Value mean() {
    final sum = values.reduce((a, b) => a + b);
    return sum * (1.0 / values.length);
  }

  ValueVector softmax() {
    final max = values.map((v) => v.data).reduce((a, b) => a > b ? a : b);
    final exps =
        values.map((v) => math.exp(v.data - max)).toList(); // Using Math.exp
    final sum = exps.reduce((a, b) => a + b);
    return ValueVector(exps.map((expVal) => Value(expVal / sum)).toList());
  }

  ValueVector sigmoid() {
    return ValueVector(
        List.generate(values.length, (int index) => values[index].sigmoid()));
  }

  Value crossEntropy(ValueVector target) {
    assert(target.values.length == values.length);
    Value loss = Value(0.0);

    for (int i = 0; i < values.length; i++) {
      loss += -target.values[i].data * math.log(values[i].data);
    }
    return loss;
  }

  @override
  String toString() => values.map((v) => v.toString()).join(', ');
}

// extension ValueActivations on Value {
//   Value relu() {
//     final out = Value(data < 0 ? 0.0 : data, [this], 'ReLU');
//     out._backward = () {
//       grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
//     };
//     return out;
//   }

//   Value sigmoid() {
//     final s = 1.0 / (1.0 + exp(-data));
//     final out = Value(s, [this], 'Sigmoid');
//     out._backward = () {
//       grad += s * (1.0 - s) * out.grad;
//     };
//     return out;
//   }
// }
