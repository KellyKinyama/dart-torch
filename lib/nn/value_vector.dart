import "dart:math" as math;
import "dart:typed_data";
import "value.dart";

class ValueVector {
  final List<Value> values;

  ValueVector(this.values);

  factory ValueVector.fromFloat32List(Float32List data) {
    return ValueVector(List.generate(data.length, (i) => Value(data[i])));
  }
  factory ValueVector.fromUint8List(Uint8List data) {
    return ValueVector(
        List.generate(data.length, (i) => Value(data[i].toDouble())));
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
  // ValueVector operator /(Value other) =>
  //     ValueVector(values.map((v) => v / other).toList());

  // Add a Value scalar to each element
  // ValueVector operator *(Value other) =>
  //     ValueVector(values.map((v) => v * other).toList());

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
      loss += (-target.values[i]) * (values[i].log());
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

  // Add these to your ValueVector class

  /// Unified multiplication operator
  /// Handles:
  /// 1. Vector * Value (Scalar)
  /// 2. Vector * ValueVector (Element-wise/Hadamard)
  ValueVector operator *(dynamic other) {
    if (other is Value) {
      // Original logic: Multiply every element by the same scalar
      return ValueVector(values.map((v) => v * other).toList());
    } else if (other is ValueVector) {
      // AFT requirement: Element-wise product
      assert(
          values.length == other.values.length, 'Vector dimensions must match');
      return ValueVector(
          List.generate(values.length, (i) => values[i] * other.values[i]));
    }
    throw UnimplementedError(
        "Multiplication not supported for: ${other.runtimeType}");
  }

  /// Unified division operator
  ValueVector operator /(dynamic other) {
    if (other is Value) {
      return ValueVector(values.map((v) => v / other).toList());
    } else if (other is ValueVector) {
      // Useful for AFT context normalization
      assert(
          values.length == other.values.length, 'Vector dimensions must match');
      return ValueVector(
          List.generate(values.length, (i) => values[i] / other.values[i]));
    }
    throw UnimplementedError(
        "Division not supported for: ${other.runtimeType}");
  }

  @override
  String toString() => "[${values.map((v) => v.toString()).join(', ')}]";

  /// Generates a vector of [length] with random values between [-scale, scale].
  /// Useful for initializing embeddings and weights.
  factory ValueVector.random(int length, {double scale = 0.02}) {
    final rand = math.Random();
    return ValueVector(List.generate(
      length,
      (_) => Value((rand.nextDouble() * 2 - 1) * scale),
    ));
  }

  /// Generates a vector of [length] filled with a specific [constant] value.
  factory ValueVector.fill(int length, double constant) {
    return ValueVector(List.generate(length, (_) => Value(constant)));
  }

  /// Returns the number of elements in the vector.
  int get length => values.length;

  /// Accesses a single Value by index.
  Value operator [](int index) => values[index];
}
