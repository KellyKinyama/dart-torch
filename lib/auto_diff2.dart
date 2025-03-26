import 'dart:math' as math;

class Value {
  double data; // <-- now mutable
  double grad = 0.0;

  late void Function() _backward;
  final Set<Value> _prev;
  final String _op;

  Value(this.data, [Set<Value>? children, this._op = ''])
      : _prev = children ?? {} {
    _backward = () {};
  }

  Value operator +(dynamic other) {
    final otherValue = other is Value ? other : Value(other.toDouble());
    final out = Value(data + otherValue.data, {this, otherValue}, '+');

    out._backward = () {
      grad += out.grad;
      otherValue.grad += out.grad;
    };

    return out;
  }

  Value operator *(dynamic other) {
    final otherValue = other is Value ? other : Value(other.toDouble());
    final out = Value(data * otherValue.data, {this, otherValue}, '*');

    out._backward = () {
      grad += otherValue.data * out.grad;
      otherValue.grad += data * out.grad;
    };

    return out;
  }

  void setData(double newData) {
    data = newData;
  }

  Value operator -() => this * -1;

  Value operator -(dynamic other) => this + (-_toValue(other));
  Value operator /(dynamic other) => this * _toValue(other).pow(-1);

  Value pow(num exponent) {
    final out = Value(data.pow(exponent), {this}, '**$exponent');

    out._backward = () {
      grad += (exponent * data.pow(exponent - 1)) * out.grad;
    };

    return out;
  }

  Value sigmoid() {
    final s = 1.0 / (1.0 + math.exp(-data));
    final out = Value(s, {this}, 'Sigmoid');
    out._backward = () {
      grad += s * (1.0 - s) * out.grad;
    };
    return out;
  }

  Value relu() {
    final out = Value(data < 0 ? 0.0 : data, {this}, 'ReLU');

    out._backward = () {
      grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
    };

    return out;
  }

  // Value relu() {
  //   final out = Value(data < 0 ? 0.0 : data, [this], 'ReLU');
  //   out._backward = () {
  //     grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
  //   };
  //   return out;
  // }

  

  void backward() {
    final topo = <Value>[];
    final visited = <Value>{};

    void buildTopo(Value v) {
      if (!visited.contains(v)) {
        visited.add(v);
        for (final child in v._prev) {
          buildTopo(child);
        }
        topo.add(v);
      }
    }

    buildTopo(this);
    grad = 1.0;

    final reversedTopo = topo.reversed;

    for (final v in reversedTopo) {
      v._backward();
    }
  }

  // @override
  // String toString() => 'Value(data=$data, grad=$grad, op=$_op, prev=$_prev)';

  @override
  String toString() => 'Value(data=$data, grad=$grad, op=$_op)';

  // Helpers
  static Value _toValue(dynamic x) => x is Value ? x : Value(x.toDouble());
}

extension Power on double {
  double pow(num exponent) => MathHelper.pow(this, exponent);
}

class MathHelper {
  static double pow(double base, num exponent) => exponent is int
      ? _intPow(base, exponent)
      : base.toDouble().pow(exponent.toDouble());

  static double _intPow(double base, int exponent) {
    if (exponent == 0) return 1.0;
    double result = 1.0;
    int exp = exponent.abs();
    for (int i = 0; i < exp; i++) {
      result *= base;
    }
    return exponent < 0 ? 1.0 / result : result;
  }
}

// void main() {
//   // final a = Value(2.0);
//   // final b = Value(3.0);
//   // final c = a * b + a;
//   // c.backward();
//   // print('$a, $b, $c');
//   final x = Value(2.0);
//   final A = Value(3.0);
//   final y = A * x;
//   final Y = Value(10.0);
//   final err = Y - y;
//   // print('$x, $A, $y');
//   err.backward();
//   print(' x: $x\r\n A: $A\r\n y: $y\r\n Y: $Y');
//   print("Error: $err");
// }

// void main() {
//   final x = Value(2.0);
//   final A = Value(1.0); // Let's start with a poor initial guess
//   final Y = Value(10.0);

//   const learningRate = 0.1;
//   const epochs = 20;

//   for (var i = 0; i < epochs; i++) {
//     // Forward pass
//     final y = A * x;
//     final err = Y - y;

//     // Reset gradients
//     A.grad = 0;
//     x.grad = 0;

//     // Backward pass
//     err.backward();

//     // Print before the update
//     print('Epoch $i');
//     print('  Error: ${err.data}');
//     print('  A: ${A.data}, A.grad: ${A.grad}');

//     // Gradient descent update (we're only updating A here)
//     final newA = A.data - learningRate * A.grad;

//     // Update A manually (you could implement `.update(grad)` method later)
//     A.grad = 0; // clear the old grad
//     // Here we mutate A’s internal state by replacing it (mutable A is better but needs Value redesign)
//     // But for now, just reassign A if you redesign to make it immutable

//     // In-place update workaround (for simplicity):
//     // Since `A.data` is final, we'd typically need a wrapper class for in-place updates.
//     // For now, re-run with a new `A` each epoch, or refactor `Value` to make `data` mutable.

//     // ❗ Let's refactor `data` in `Value` to be mutable instead.
//     A._setData(newA);

//     print('  Updated A: ${A.data}');
//     print('');
//   }
// }

// void main() {
//   // Inputs
//   final x1 = Value(1.0);
//   final x2 = Value(2.0);
//   final x3 = Value(3.0);
//   final x4 = Value(4.0);

//   // Initial weights (parameters to learn)
//   final A1 = Value(0.1);
//   final A2 = Value(0.1);
//   final A3 = Value(0.1);
//   final A4 = Value(0.1);

//   // Target output
//   final Y = Value(30.0); // you can change this to simulate a dataset

//   const learningRate = 0.01;
//   const epochs = 50;

//   for (var i = 0; i < epochs; i++) {
//     // Forward pass
//     final y = A1 * x1 + A2 * x2 + A3 * x3 + A4 * x4;
//     final err = Y - y;

//     // Reset gradients
//     for (var p in [A1, A2, A3, A4]) {
//       p.grad = 0;
//     }

//     // Backward pass
//     err.backward();

//     // Update weights using gradient descent
//     A1._setData(A1.data - learningRate * A1.grad);
//     A2._setData(A2.data - learningRate * A2.grad);
//     A3._setData(A3.data - learningRate * A3.grad);
//     A4._setData(A4.data - learningRate * A4.grad);

//     // Print loss and weights
//     print('Epoch $i');
//     print('  y: ${y.data}, Error: ${err.data}');
//     print('  A1: ${A1.data}, A2: ${A2.data}, A3: ${A3.data}, A4: ${A4.data}');
//     print('');
//   }
// }

void main() {
  // Inputs
  final x1 = Value(1.0);
  final x2 = Value(2.0);
  final x3 = Value(3.0);
  final x4 = Value(4.0);

  // Learnable parameters (weights)
  final A1 = Value(0.1);
  final A2 = Value(0.1);
  final A3 = Value(0.1);
  final A4 = Value(0.1);

  // Target output
  final Y = Value(30.0); // Expected target

  const learningRate = 0.01;
  const epochs = 50;

  for (var i = 0; i < epochs; i++) {
    // Forward pass: y = A1*x1 + A2*x2 + A3*x3 + A4*x4
    final y = A1 * x1 + A2 * x2 + A3 * x3 + A4 * x4;

    // Loss function: Mean Squared Error = (y - Y)^2
    final diff = y - Y;
    final loss = diff * diff; // This is MSE for 1 sample

    // Reset gradients
    for (var p in [A1, A2, A3, A4]) {
      p.grad = 0;
    }

    // Backward pass
    loss.backward();

    // Update weights using gradient descent
    A1.setData(A1.data - learningRate * A1.grad);
    A2.setData(A2.data - learningRate * A2.grad);
    A3.setData(A3.data - learningRate * A3.grad);
    A4.setData(A4.data - learningRate * A4.grad);

    // Print progress
    print('Epoch $i');
    print('  y: ${y.data}, Loss (MSE): ${loss.data}');
    print('  A1: ${A1.data}, A2: ${A2.data}, A3: ${A3.data}, A4: ${A4.data}\n');
  }
}
