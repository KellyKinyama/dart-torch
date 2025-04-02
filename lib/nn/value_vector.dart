import "dart:math" as math;
import "dart:typed_data";
import "value.dart";

class ValueVector {
  final List<Value> values;

  ValueVector(this.values);

  factory ValueVector.fromFloat32List(Float32List data) {
    return ValueVector(List.generate(data.length, (i) => Value(data[i])));
  }
  factory ValueVector.fromDoubleList(List<double> data) {
    return ValueVector(List.generate(data.length, (i) => Value(data[i])));
  }

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
  ValueVector operator +(dynamic other) {
    if (other is Value) {
      return ValueVector(values.map((v) => v + other).toList());
    }
    if (other is ValueVector) {
      //  ValueVector operator +(ValueVector other) {
      if (values.length != other.values.length) {
        throw ArgumentError('Vector dimensions must match for addition');
      }
      return ValueVector(
          List.generate(values.length, (i) => values[i] + other.values[i]));
    }
    throw UnimplementedError(
        "Operation + not supported for: ${other.runtimeType}");
  }

  // Add a Value scalar to each element
  ValueVector operator /(Value other) =>
      ValueVector(values.map((v) => v / other).toList());

  // Add a Value scalar to each element
  ValueVector operator *(Value other) =>
      ValueVector(values.map((v) => v * other).toList());

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

  Value crossEntropy(ValueVector target) {
    assert(target.values.length == values.length);
    Value loss = Value(0.0);

    for (int i = 0; i < values.length; i++) {
      loss += -target.values[i].data * (values[i].log().data);
    }
    return loss;
  }

  ValueVector sigmoid() {
    return ValueVector(
        List.generate(values.length, (int index) => values[index].sigmoid()));
  }

  ValueVector softmax() {
    return ValueVector(Value.softmax(values));
  }

  ValueVector reLU() {
    return ValueVector(
        List.generate(values.length, (int index) => values[index].relu()));
  }

  @override
  String toString() => values.map((v) => v.toString()).join(', ');
}
