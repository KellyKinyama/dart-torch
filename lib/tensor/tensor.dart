import 'dart:math' as math;
import 'dart:typed_data';

/// Standalone Tanh helper to replace missing math.tanh in Dart
double _mathTanh(double x) {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  double e2x = math.exp(2 * x);
  return (e2x - 1) / (e2x + 1);
}

class Tensor {
  final Float32List data;
  final Float32List grad;
  final List<int> shape;
  final int length;
  final Set<Tensor> _prev;

  void Function() _backward = () {};

  Tensor(this.shape, {Iterable<Tensor>? children})
      : length = shape.isEmpty ? 0 : shape.reduce((a, b) => a * b),
        data = Float32List(shape.isEmpty ? 0 : shape.reduce((a, b) => a * b)),
        grad = Float32List(shape.isEmpty ? 0 : shape.reduce((a, b) => a * b)),
        _prev = children?.toSet() ?? {};

  // --- Initializers ---

  factory Tensor.fill(List<int> shape, double val) {
    final t = Tensor(shape);
    t.data.fillRange(0, t.length, val);
    return t;
  }

  factory Tensor.random(List<int> shape) {
    final t = Tensor(shape);
    final rand = math.Random();
    double nIn = shape[0].toDouble();
    double limit = math.sqrt(1.0 / nIn);
    for (int i = 0; i < t.length; i++) {
      t.data[i] = (rand.nextDouble() * 2 - 1) * limit;
    }
    return t;
  }

  static Tensor xavier(List<int> shape) {
    final t = Tensor(shape);
    final int nIn = shape[0];
    final int nOut = shape[shape.length - 1];
    final double limit = math.sqrt(6.0 / (nIn + nOut));
    final rand = math.Random();
    for (int i = 0; i < t.length; i++) {
      t.data[i] = (rand.nextDouble() * 2 - 1) * limit;
    }
    return t;
  }

  // --- Basic Operators ---

  Tensor operator +(dynamic other) {
    final out = Tensor(shape, children: {this});
    if (other is Tensor) {
      out._prev.add(other);
      for (int i = 0; i < length; i++) {
        out.data[i] = data[i] + other.data[i % other.length];
      }
      out.onBackward = () {
        for (int i = 0; i < length; i++) {
          grad[i] += out.grad[i];
          other.grad[i % other.length] += out.grad[i];
        }
      };
    } else if (other is num) {
      final double scalar = other.toDouble();
      for (int i = 0; i < length; i++) out.data[i] = data[i] + scalar;
      out.onBackward = () {
        for (int i = 0; i < length; i++) grad[i] += out.grad[i];
      };
    }
    return out;
  }

  Tensor operator *(dynamic other) {
    final out = Tensor(shape, children: {this});
    if (other is Tensor) {
      out._prev.add(other);
      for (int i = 0; i < length; i++) out.data[i] = data[i] * other.data[i];
      out.onBackward = () {
        for (int i = 0; i < length; i++) {
          grad[i] += other.data[i] * out.grad[i];
          other.grad[i] += data[i] * out.grad[i];
        }
      };
    } else if (other is num) {
      final double scalar = other.toDouble();
      for (int i = 0; i < length; i++) out.data[i] = data[i] * scalar;
      out.onBackward = () {
        for (int i = 0; i < length; i++) grad[i] += scalar * out.grad[i];
      };
    }
    return out;
  }

  Tensor operator -() {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++) out.data[i] = -data[i];
    out.onBackward = () {
      for (int i = 0; i < length; i++) grad[i] -= out.grad[i];
    };
    return out;
  }

  Tensor operator -(dynamic other) {
    if (other is num) return this + (-other.toDouble());
    if (other is Tensor) return this + (-other);
    throw ArgumentError("Subtraction only supported for Tensor or num");
  }

  Tensor operator /(dynamic other) {
    if (other is num) {
      if (other == 0) throw UnsupportedError("Division by zero scalar.");
      // Reuse multiplication logic with reciprocal
      return this * (1.0 / other.toDouble());
    } else if (other is Tensor) {
      for (int i = 0; i < other.length; i++) {
        if (other.data[i] == 0) {
          throw UnsupportedError("Division by zero in divisor Tensor.");
        }
      }
      final out = Tensor(shape, children: {this, other});
      for (int i = 0; i < length; i++) out.data[i] = data[i] / other.data[i];
      out.onBackward = () {
        for (int i = 0; i < length; i++) {
          double den = other.data[i];
          grad[i] += out.grad[i] / den;
          other.grad[i] += out.grad[i] * (-data[i] / (den * den));
        }
      };
      return out;
    }
    throw ArgumentError("Division only supported for Tensor or num");
  }

  // --- Matrix Multiplication ---

  Tensor matmul(Tensor other) {
    assert(shape[1] == other.shape[0]);
    int M = shape[0], K = shape[1], N = other.shape[1];
    final out = Tensor([M, N], children: {this, other});
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
          out.data[i * N + j] += data[i * K + k] * other.data[k * N + j];
        }
      }
    }
    out.onBackward = () {
      for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
          for (int j = 0; j < N; j++) {
            double og = out.grad[i * N + j];
            grad[i * K + k] += other.data[k * N + j] * og;
            other.grad[k * N + j] += data[i * K + k] * og;
          }
        }
      }
    };
    return out;
  }

  // --- Functions ---

  Tensor abs() {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++) out.data[i] = data[i].abs();
    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        if (data[i] > 0)
          grad[i] += out.grad[i];
        else if (data[i] < 0) grad[i] -= out.grad[i];
      }
    };
    return out;
  }

  Tensor pow(num exponent) {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++)
      out.data[i] = math.pow(data[i], exponent).toDouble();
    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        grad[i] += (exponent * math.pow(data[i], exponent - 1).toDouble()) *
            out.grad[i];
      }
    };
    return out;
  }

  Tensor exp() {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++) out.data[i] = math.exp(data[i]);
    out.onBackward = () {
      for (int i = 0; i < length; i++) grad[i] += out.data[i] * out.grad[i];
    };
    return out;
  }

  Tensor sqrt() {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++) {
      if (data[i] < 0)
        throw UnsupportedError("Square root of negative number.");
      out.data[i] = math.sqrt(data[i]);
    }
    out.onBackward = () {
      for (int i = 0; i < length; i++)
        grad[i] += (0.5 / (out.data[i] + 1e-9)) * out.grad[i];
    };
    return out;
  }

  // --- Activations ---

  Tensor relu() {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++) out.data[i] = math.max(0, data[i]);
    out.onBackward = () {
      for (int i = 0; i < length; i++) if (data[i] > 0) grad[i] += out.grad[i];
    };
    return out;
  }

  Tensor sigmoid() {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++)
      out.data[i] = 1.0 / (1.0 + math.exp(-data[i]));
    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        double s = out.data[i];
        grad[i] += s * (1.0 - s) * out.grad[i];
      }
    };
    return out;
  }

  Tensor gelu() {
    final out = Tensor(shape, children: {this});
    final s2p = math.sqrt(2 / math.pi);
    for (int i = 0; i < length; i++) {
      double x = data[i];
      out.data[i] =
          0.5 * x * (1 + _mathTanh(s2p * (x + 0.044715 * math.pow(x, 3))));
    }
    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        double x = data[i];
        double cdf =
            0.5 * (1.0 + _mathTanh(s2p * (x + 0.044715 * math.pow(x, 3))));
        double pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x);
        grad[i] += (cdf + x * pdf) * out.grad[i];
      }
    };
    return out;
  }

  Tensor softmax() {
    final out = Tensor(shape, children: {this});
    double maxVal = data.reduce(math.max);
    double sumExp = 0;
    for (int i = 0; i < length; i++) {
      out.data[i] = math.exp(data[i] - maxVal);
      sumExp += out.data[i];
    }
    for (int i = 0; i < length; i++) out.data[i] /= (sumExp + 1e-9);
    out.onBackward = () {
      double dot = 0;
      for (int i = 0; i < length; i++) dot += out.data[i] * out.grad[i];
      for (int i = 0; i < length; i++)
        grad[i] += out.data[i] * (out.grad[i] - dot);
    };
    return out;
  }

  // --- Slicing ---

  Tensor slice(int startOffset, int endOffset) {
    final int sliceLength = endOffset - startOffset;
    final out = Tensor([sliceLength], children: {this});
    for (int i = 0; i < sliceLength; i++) out.data[i] = data[startOffset + i];
    out.onBackward = () {
      for (int i = 0; i < sliceLength; i++)
        grad[startOffset + i] += out.grad[i];
    };
    return out;
  }

  Tensor slice2D(int r, int c) {
    final out = Tensor([r, c], children: {this});
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        out.data[i * c + j] = data[i * shape[1] + j];
      }
    }
    out.onBackward = () {
      for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
          grad[i * shape[1] + j] += out.grad[i * c + j];
        }
      }
    };
    return out;
  }

  // --- Training Helpers ---

  Tensor mseLoss(Tensor target) {
    final out = Tensor([1], children: {this, target});
    double diffSum = 0;
    for (int i = 0; i < length; i++) {
      double d = data[i] - target.data[i];
      diffSum += d * d;
    }
    out.data[0] = diffSum / length;
    out.onBackward = () {
      double factor = 2.0 / length;
      for (int i = 0; i < length; i++) {
        grad[i] += factor * (data[i] - target.data[i]) * out.grad[0];
      }
    };
    return out;
  }

  set onBackward(void Function() func) => _backward = func;

  void _runBackward() => _backward();

  void backward() {
    final topo = <Tensor>[];
    final visited = <Tensor>{};
    void build(Tensor t) {
      if (visited.add(t)) {
        for (final p in t._prev) build(p);
        topo.add(t);
      }
    }

    build(this);
    grad.fillRange(0, length, 1.0);
    for (final t in topo.reversed) t._runBackward();
  }

  void zeroGrad() => grad.fillRange(0, length, 0.0);
}

void main() {
  print('--- Example 1: Basic Arithmetic ---');
  Tensor a = Tensor.fill([1, 1], 2.0);
  Tensor b = Tensor.fill([1, 1], 3.0);
  final c = a + b;
  c.backward();
  print('c: ${c.data[0]}  // Expected: 5.0');
  print('a: ${a.grad[0]}  // Expected: 1.0');
  print('b: ${b.grad[0]}  // Expected: 1.0');

  print('\n--- Example 2: Multiplication ---');
  a = Tensor.fill([1, 1], 2.0);
  b = Tensor.fill([1, 1], 3.0);
  final d = a * b;
  d.backward();
  print('d: ${d.data[0]}  // Expected: 6.0');
  print('a: ${a.grad[0]}  // Expected: 3.0');
  print('b: ${b.grad[0]}  // Expected: 2.0');

  print('\n--- Example 3: Polynomial y = x^2 + 3x + 1 ---');
  final x1 = Tensor.fill([1, 1], 2.0);
  final y1 = (x1 * x1) + (x1 * 3.0) + 1.0;
  y1.backward();
  print('y1: ${y1.data[0]}  // Expected: 11.0');
  print('x1: ${x1.grad[0]}  // Expected: 7.0');

  print('\n--- Example 4: Power y = x^3 ---');
  final x2 = Tensor.fill([1, 1], 2.0);
  final y2 = x2.pow(3);
  y2.backward();
  print('y2: ${y2.data[0]}  // Expected: 8.0');
  print('x2: ${x2.grad[0]}  // Expected: 12.0');

  print('\n--- Example 5: Negative and Division y = -a / b ---');
  final a2 = Tensor.fill([1, 1], 4.0);
  final b2 = Tensor.fill([1, 1], 2.0);
  final y3 = (-a2) / b2;
  y3.backward();
  print('y3: ${y3.data[0]}  // Expected: -2.0');
  print('a2: ${a2.grad[0]}  // Expected: -0.5');
  print('b2: ${b2.grad[0]}  // Expected: 1.0');

  print('\n--- Example 6: Sigmoid Activation ---');
  final x3 = Tensor.fill([1, 1], 1.0);
  final y4 = x3.sigmoid();
  y4.backward();
  print('y4: ${y4.data[0].toStringAsFixed(4)}  // Expected ≈ 0.7311');
  print('x3: ${x3.grad[0].toStringAsFixed(4)}  // Expected grad ≈ 0.1966');

  print('\n--- Example 7: ReLU Activation (x < 0) ---');
  final x4 = Tensor.fill([1, 1], -2.0);
  final y5 = x4.relu();
  y5.backward();
  print('y5: ${y5.data[0]}  // Expected: 0.0');
  print('x4: ${x4.grad[0]}  // Expected: 0.0');

  print('\n--- Example 8: ReLU Activation (x > 0) ---');
  final x5 = Tensor.fill([1, 1], 3.0);
  final y6 = x5.relu();
  y6.backward();
  print('y6: ${y6.data[0]}  // Expected: 3.0');
  print('x5: ${x5.grad[0]}  // Expected: 1.0');

  print('\n--- Example 9: Composite Expression y = sigmoid(a * x + b) * c ---');
  final xc = Tensor.fill([1, 1], 2.0);
  final ac = Tensor.fill([1, 1], 3.0);
  final bc = Tensor.fill([1, 1], 1.0);
  final cc = Tensor.fill([1, 1], -1.0);
  final yc = ((ac * xc + bc).sigmoid()) * cc;
  yc.backward();
  print('yc: ${yc.data[0].toStringAsFixed(5)}');
  print('xc: ${xc.grad[0].toStringAsFixed(5)}  // Expected ≈ -0.00273');
  print('ac: ${ac.grad[0].toStringAsFixed(5)}  // Expected ≈ -0.00182');
  print('bc: ${bc.grad[0].toStringAsFixed(5)}  // Expected ≈ -0.00091');
  print('cc: ${cc.grad[0].toStringAsFixed(5)}  // Expected ≈ 0.99909');

  print('\n--- Example 10: Quadratic Loss = (yTrue - yPred)^2 ---');
  final x6 = Tensor.fill([1, 1], 2.0);
  final w = Tensor.fill([1, 1], 3.0);
  final yPred = w * x6;
  final yTrue = Tensor.fill([1, 1], 10.0);
  final loss = (yTrue - yPred).pow(2);
  loss.backward();
  print('loss: ${loss.data[0]}  // Expected: 16.0');
  print('x6: ${x6.grad[0]}  // Expected: -24');
  print('w : ${w.grad[0]}  // Expected: -16');

  print('\n--- Example 11: Chain Rule ---');
  final x7 = Tensor.fill([1, 1], 2.0);
  final y7 = x7 * 3.0;
  final z7 = y7 + 5.0;
  final out7 = z7.pow(2);
  out7.backward();
  print('out7: ${out7.data[0]}  // Expected: 121.0');
  print('x7: ${x7.grad[0]}  // Expected: 66.0');

  print('\n--- Example 12: Simple Addition with Negation ---');
  final x8 = Tensor.fill([1, 1], 5.0);
  final y8 = Tensor.fill([1, 1], 3.0);
  final z8 = -(x8 + y8);
  z8.backward();
  print('z8: ${z8.data[0]}  // Expected: -8.0');
  print('x8: ${x8.grad[0]}  // Expected: -1.0');
  print('y8: ${y8.grad[0]}  // Expected: -1.0');

  print('\n--- Example 13: Chain of Operations (x + 2) * (y - 1) ---');
  final x9 = Tensor.fill([1, 1], 4.0);
  final y9 = Tensor.fill([1, 1], 6.0);
  final z9 = (x9 + 2.0) * (y9 - 1.0);
  z9.backward();
  print('z9: ${z9.data[0]}  // Expected: 30.0');
  print('x9: ${x9.grad[0]}  // Expected: 5.0');
  print('y9: ${y9.grad[0]}  // Expected: 6.0');

  print('\n--- Example 14: More Complex Expression ---');
  final x10 = Tensor.fill([1, 1], 1.0);
  final y10 = x10 * 2.0;
  final z10 = (y10 + 3.0).pow(2);
  final out10 = z10 / 4.0;
  out10.backward();
  print('out10: ${out10.data[0]}  // Expected: 6.25');
  print('x10: ${x10.grad[0]}  // Expected: 5.0');
}
