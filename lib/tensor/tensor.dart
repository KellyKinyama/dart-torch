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

  Tensor operator *(Tensor other) {
    final out = Tensor(shape, children: {this, other});
    for (int i = 0; i < length; i++) out.data[i] = data[i] * other.data[i];
    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        grad[i] += other.data[i] * out.grad[i];
        other.grad[i] += data[i] * out.grad[i];
      }
    };
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

  Tensor operator -(Tensor other) => this + (-other);

  /// Division with explicit Error handling for Division by Zero
  Tensor operator /(dynamic other) {
    if (other is num) {
      if (other == 0) throw UnsupportedError("Division by zero scalar.");
      return this * Tensor.fill(shape, 1.0 / other.toDouble());
    } else if (other is Tensor) {
      // Check for any zero values in the divisor tensor
      for (int i = 0; i < other.length; i++) {
        if (other.data[i] == 0) {
          throw UnsupportedError(
              "Division by zero detected at index $i in divisor Tensor.");
        }
      }

      final out = Tensor(shape, children: {this, other});
      for (int i = 0; i < length; i++) {
        out.data[i] = data[i] / other.data[i];
      }
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
        throw UnsupportedError("Square root of negative number at index $i");
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
    if (sumExp == 0)
      throw UnsupportedError("Softmax sum is zero (input values too small).");
    for (int i = 0; i < length; i++) out.data[i] /= sumExp;
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
    final out = Tensor([1, sliceLength], children: {this});
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

  set onBackward(void Function() func) {
    _backward = func;
  }

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
