import 'dart:math' as math;

double tanH(double x) {
  return (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1);
}

class Value {
  double data; // Forward pass value
  double grad = 0.0; // Gradient from backward pass

  late void Function() _backward;
  final Set<Value> _prev;
  final String _op;

  Value(this.data, [Set<Value>? children, this._op = ''])
      : _prev = children ?? {} {
    _backward = () {};
  }

  // Operators

  Value operator +(dynamic other) {
    final o = Value.toValue(other);
    final out = Value(data + o.data, {this, o}, '+');
    out._backward = () {
      grad += out.grad;
      o.grad += out.grad;
    };
    return out;
  }

  Value operator *(dynamic other) {
    final o = Value.toValue(other);
    final out = Value(data * o.data, {this, o}, '*');
    out._backward = () {
      grad += o.data * out.grad; // Correct gradient for this
      o.grad += data * out.grad; // Correct gradient for o
    };
    return out;
  }

  Value operator -() => this * -1;
  Value operator -(dynamic other) => this + (-Value.toValue(other));

  Value operator /(dynamic other) {
    final o = Value.toValue(other);
    if (o.data == 0) {
      throw Exception("Division by zero error.");
    }
    final out = Value(data / o.data, {this, o}, '/');
    out._backward = () {
      grad += (1 / o.data) * out.grad; // Correct for this
      o.grad += (-data / (o.data * o.data)) * out.grad; // Correct for o
    };
    return out;
  }

  Value pow(num exponent) {
    final result = math.pow(data, exponent).toDouble();
    final out = Value(result, {this}, '**$exponent');
    out._backward = () {
      grad += (exponent * math.pow(data, exponent - 1).toDouble()) * out.grad;
    };
    return out;
  }

  // Activation functions
  Value relu() {
    final out = Value(data < 0 ? 0.0 : data, {this}, 'ReLU');
    out._backward = () {
      grad += (out.data > 0 ? 1.0 : 0.0) * out.grad;
    };

    return out;
  }

  Value sigmoid() {
    final s = 1.0 / (1.0 + math.exp(-data));
    final out = Value(s, {this}, 'Sigmoid');
    out._backward = () {
      grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  Value tanh() {
    final t = (math.exp(2 * data) - 1) / (math.exp(2 * data) + 1);
    // final t = math.tanh(data);
    final out = Value(t, {this}, 'Tanh');
    out._backward = () {
      grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  Value elu(double alpha) {
    final out = data >= 0
        ? Value(data, {this}, 'ELU')
        : Value(alpha * (math.exp(data) - 1), {this}, 'ELU');
    out._backward = () {
      grad += (out.data >= 0 ? 1.0 : alpha * math.exp(data)) * out.grad;
    };
    return out;
  }

  Value gelu() {
    final cdf = 0.5 * (1 + tanH(data / math.sqrt(2)));
    final out = Value(data * cdf, {this}, 'GELU');
    out._backward = () {
      final derivative = 0.5 * (1 + tanH(data / math.sqrt(2))) +
          (data * (1 - tanH(data / math.sqrt(2)) * tanH(data / math.sqrt(2)))) /
              (2 * math.sqrt(2 * math.pi));
      grad += derivative * out.grad;
    };
    return out;
  }

  // Zero gradients recursively
  void zeroGrad() {
    final visited = <Value>{};
    // ignore: no_leading_underscores_for_local_identifiers
    void _reset(Value v) {
      if (!visited.contains(v)) {
        visited.add(v);
        v.grad = 0.0;
        for (final child in v._prev) {
          _reset(child);
        }
      }
    }

    _reset(this);
  }

  Value exp() {
    final expVal = math.exp(data);
    final out = Value(expVal, {this}, 'exp');
    out._backward = () {
      grad += expVal * out.grad;
    };
    return out;
  }

  Value log() {
    final out = Value(math.log(data), {this}, 'log');

    out._backward = () {
      grad += (1 / data) * out.grad;
    };
    return out;
  }

  static List<Value> softmax(List<Value> inputs) {
    double maxVal = inputs.map((n) => n.data).reduce(math.max);
    List<Value> expVals =
        inputs.map((n) => (n - Value(maxVal, {n}, "softmax")).exp()).toList();
    Value sumExp = expVals.reduce((a, b) => a + b);
    List<Value> softmaxOut = expVals.map((n) => n / sumExp).toList();

    return softmaxOut;
  }

  Value sqrt() {
    if (data < 0) {
      throw Exception("Square root of a negative number is undefined.");
    }
    final result = math.sqrt(data);
    final out = Value(result, {this}, 'sqrt');
    out._backward = () {
      grad += (0.5 / result) * out.grad;
    };
    return out;
  }

  // Backpropagation
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
    for (final v in topo.reversed) {
      v._backward();
    }
  }

  void setData(double newData) => data = newData;

  @override
  String toString() => 'Value(data=$data, grad=$grad, op=$_op)';

  static Value toValue(dynamic x) => x is Value ? x : Value(x.toDouble());
}

void main() {
  print('--- Example 1: Basic Arithmetic ---');
  Value a = Value(2.0);
  Value b = Value(3.0);
  final c = a + b;
  c.backward();
  print('c: $c  // Expected: data=5.0');
  print('a: $a  // Expected: grad=1.0');
  print('b: $b  // Expected: grad=1.0');

  print('\n--- Example 2: Multiplication ---');
  a = Value(2.0);
  b = Value(3.0);
  final d = a * b;
  d.backward();
  print('d: $d  // Expected: data=6.0');
  print('a: $a  // Expected: grad=3.0');
  print('b: $b  // Expected: grad=2.0');

  print('\n--- Example 3: Polynomial y = x^2 + 3x + 1 ---');
  final x1 = Value(2.0);
  final y1 = x1 * x1 + x1 * 3.0 + 1;
  y1.backward();
  print('y1: $y1  // Expected: data=11.0');
  print('x1: $x1  // Expected: grad=7.0');

  print('\n--- Example 4: Power y = x^3 ---');
  final x2 = Value(2.0);
  final y2 = x2.pow(3);
  y2.backward();
  print('y2: $y2  // Expected: data=8.0');
  print('x2: $x2  // Expected: grad=12.0');

  print('\n--- Example 5: Negative and Division y = -a / b ---');
  final a2 = Value(4.0);
  final b2 = Value(2.0);
  final y3 = -a2 / b2;
  y3.backward();
  print('y3: $y3  // Expected: data=-2.0');
  print('a2: $a2  // Expected: grad=-0.5');
  print('b2: $b2  // Expected: grad=1.0');

  print('\n--- Example 6: Sigmoid Activation ---');
  final x3 = Value(1.0);
  final y4 = x3.sigmoid();
  y4.backward();
  print('y4: $y4  // Expected ≈ 0.7311');
  print('x3: $x3  // Expected grad ≈ 0.1966');

  print('\n--- Example 7: ReLU Activation (x < 0) ---');
  final x4 = Value(-2.0);
  final y5 = x4.relu();
  y5.backward();
  print('y5: $y5  // Expected data=0.0');
  print('x4: $x4  // Expected grad=0.0');

  print('\n--- Example 8: ReLU Activation (x > 0) ---');
  final x5 = Value(3.0);
  final y6 = x5.relu();
  y6.backward();
  print('y6: $y6  // Expected data=3.0');
  print('x5: $x5  // Expected grad=1.0');

  print('\n--- Example 9: Composite Expression y = sigmoid(a * x + b) * c ---');
  final xc = Value(2.0);
  final ac = Value(3.0);
  final bc = Value(1.0);
  final cc = Value(-1.0);
  final yc = ((ac * xc + bc).sigmoid()) * cc;
  yc.backward();
  print('yc: $yc');
  print('xc: $xc  // Expected small negative grad ≈ -0.00273');
  print('ac: $ac  // Expected ≈ -0.00182');
  print('bc: $bc  // Expected ≈ -0.00091');
  print('cc: $cc  // Expected ≈ sigmoid ≈ 0.99909');

  print('\n--- Example 10: Quadratic Loss = (yTrue - yPred)^2 ---');
  final x6 = Value(2.0);
  final w = Value(3.0);
  final yPred = w * x6;
  final yTrue = Value(10.0);
  final loss = (yTrue - yPred).pow(2);
  loss.backward();
  print('loss: $loss  // Expected=16.0');
  print('x6: $x6  // Expected grad = -24');
  print('w : $w  // Expected grad = -16');
  print('yPred: $yPred');
  print('yTrue: $yTrue');

  print('\n--- Example 11: Chain Rule ---');
  final x7 = Value(2.0);
  final y7 = x7 * 3.0;
  final z7 = y7 + 5.0;
  final out7 = z7.pow(2);
  out7.backward();
  print('out7: $out7  // Expected=121.0');
  print('x7: $x7  // Expected grad=66.0');

  print('\n--- Example 12: Simple Addition with Negation ---');
  final x8 = Value(5.0);
  final y8 = Value(3.0);
  final z8 = -(x8 + y8);
  z8.backward();
  print('z8: $z8  // Expected data=-8.0');
  print('x8: $x8  // Expected grad=-1.0');
  print('y8: $y8  // Expected grad=-1.0');

  print('\n--- Example 13: Chain of Operations (x + 2) * (y - 1) ---');
  final x9 = Value(4.0);
  final y9 = Value(6.0);
  final z9 = (x9 + 2.0) * (y9 - 1.0);
  z9.backward();
  print('z9: $z9  // Expected data=30.0');
  print('x9: $x9  // Expected grad=4.0');
  print('y9: $y9  // Expected grad=3.0');

  print('\n--- Example 14: More Complex Expression ---');
  final x10 = Value(1.0);
  final y10 = x10 * 2.0;
  final z10 = (y10 + 3.0).pow(2);
  final out10 = z10 / 4.0;
  out10.backward();
  print('out10: $out10  // Expected data=10.0');
  print('x10: $x10  // Expected grad=10.0');
}
