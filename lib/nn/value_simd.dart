import 'dart:typed_data';
import 'dart:math' as math;

class ValueSIMD {
  Float32x4 data;
  Float32x4 grad = Float32x4.zero();
  late void Function() _backward;
  final Set<ValueSIMD> _prev;
  final String _op;

  ValueSIMD(double val, [Set<ValueSIMD>? children, this._op = ''])
      : data = Float32x4.splat(val),
        _prev = children ?? {} {
    _backward = () {};
  }

  ValueSIMD.fromSIMD(Float32x4 simd, [Set<ValueSIMD>? children, this._op = ''])
      : data = simd,
        _prev = children ?? {} {
    _backward = () {};
  }

  // Addition
  ValueSIMD operator +(ValueSIMD other) {
    final out = ValueSIMD.fromSIMD(data + other.data, {this, other}, '+');
    out._backward = () {
      grad += out.grad;
      other.grad += out.grad;
    };
    return out;
  }

  // Subtraction
  ValueSIMD operator -(ValueSIMD other) {
    final out = ValueSIMD.fromSIMD(data - other.data, {this, other}, '-');
    out._backward = () {
      grad += out.grad;
      other.grad -= out.grad;
    };
    return out;
  }

  // Multiplication
  ValueSIMD operator *(ValueSIMD other) {
    final out = ValueSIMD.fromSIMD(data * other.data, {this, other}, '*');
    out._backward = () {
      grad += other.data * out.grad;
      other.grad += data * out.grad;
    };
    return out;
  }

  // Division
  ValueSIMD operator /(ValueSIMD other) {
    final divVal = data / other.data;
    final out = ValueSIMD.fromSIMD(divVal, {this, other}, '/');
    out._backward = () {
      grad += (Float32x4.splat(1.0) / other.data) * out.grad;
      other.grad -= (data / (other.data * other.data)) * out.grad;
    };
    return out;
  }

  // Exponential using SIMD
  ValueSIMD exp() {
    final expVal = data.exp();
    final out = ValueSIMD.fromSIMD(expVal, {this}, 'exp');
    out._backward = () {
      grad += expVal * out.grad;
    };
    return out;
  }

  // Sigmoid with SIMD
  ValueSIMD sigmoid() {
    final expNegX = (-data).exp();
    final sigmoidVal = Float32x4.splat(1.0) / (Float32x4.splat(1.0) + expNegX);
    final out = ValueSIMD.fromSIMD(sigmoidVal, {this}, 'sigmoid');
    out._backward = () {
      grad += sigmoidVal * (Float32x4.splat(1.0) - sigmoidVal) * out.grad;
    };
    return out;
  }

  // Tanh function using SIMD
  ValueSIMD tanh() {
    final exp2x = (data * Float32x4.splat(2.0)).exp();
    final tanhVal = (exp2x - Float32x4.splat(1.0)) / (exp2x + Float32x4.splat(1.0));
    final out = ValueSIMD.fromSIMD(tanhVal, {this}, 'tanh');
    out._backward = () {
      grad += (Float32x4.splat(1.0) - tanhVal * tanhVal) * out.grad;
    };
    return out;
  }

  // Backpropagation
  void backward() {
    grad = Float32x4.splat(1.0);
    final visited = <ValueSIMD>{};
    final stack = [this];

    while (stack.isNotEmpty) {
      final node = stack.removeLast();
      node._backward();
      visited.add(node);
      for (final child in node._prev) {
        if (!visited.contains(child)) {
          stack.add(child);
        }
      }
    }
  }

  @override
  String toString() {
    return 'ValueSIMD(data=${data.x}, grad=${grad.x}, op=$_op)';
  }
}
