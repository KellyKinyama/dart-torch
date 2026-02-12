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

  void Function()? _backward;

  Tensor(this.shape, {Iterable<Tensor>? children})
      : length = shape.isEmpty ? 0 : shape.reduce((a, b) => a * b),
        data = Float32List(shape.isEmpty ? 0 : shape.reduce((a, b) => a * b)),
        grad = Float32List(shape.isEmpty ? 0 : shape.reduce((a, b) => a * b)),
        _prev = children?.toSet() ?? {};

  // --- Initializers ---
  /// Creates a new Tensor filled with 0.0
  factory Tensor.zeros(List<int> shape, {Set<Tensor>? children}) {
    int totalLength = shape.reduce((a, b) => a * b);
    return Tensor(shape, children: children)
      ..data.fillRange(0, totalLength, 0.0);
  }

  /// Creates a new Tensor filled with 1.0
  factory Tensor.ones(List<int> shape, {Set<Tensor>? children}) {
    int totalLength = shape.reduce((a, b) => a * b);
    return Tensor(shape, children: children)
      ..data.fillRange(0, totalLength, 1.0);
  }

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

      // CAPTURE: Create a local, typed reference for the closure
      final Tensor otherTensor = other;
      final Tensor thisTensor = this;

      // 1. Forward Pass
      for (int i = 0; i < length; i++) {
        out.data[i] = data[i] + otherTensor.data[i % otherTensor.length];
      }

      // 2. Backward Pass
      out.onBackward = () {
        for (int i = 0; i < length; i++) {
          double g = out.grad[i];

          // 3. DEBUGGING: Let's remove the clipping for a moment
          // If gradients were small, clipping to 10.0 doesn't help.
          // If they were NaN, we want to know why.
          // if (g.isNaN) g = 0.0;

          // Propagate using the captured references
          thisTensor.grad[i] += g;
          otherTensor.grad[i % otherTensor.length] += g;
        }
      };
    } else if (other is num) {
      final double scalar = other.toDouble();
      final Tensor thisTensor = this;

      for (int i = 0; i < length; i++) {
        out.data[i] = data[i] + scalar;
      }

      out.onBackward = () {
        for (int i = 0; i < length; i++) {
          double g = out.grad[i];
          if (!g.isNaN) thisTensor.grad[i] += g;
        }
      };
    }
    return out;
  }

  Tensor operator *(dynamic other) {
    final out = Tensor(shape, children: {this});
    final Tensor left = this; // Capture

    if (other is Tensor) {
      final Tensor right = other; // Capture
      out._prev.add(right);
      for (int i = 0; i < length; i++) out.data[i] = data[i] * right.data[i];
      out.onBackward = () {
        for (int i = 0; i < length; i++) {
          left.grad[i] += right.data[i] * out.grad[i];
          right.grad[i] += left.data[i] * out.grad[i];
        }
      };
    } else if (other is num) {
      final double scalar = other.toDouble();
      for (int i = 0; i < length; i++) out.data[i] = data[i] * scalar;
      out.onBackward = () {
        for (int i = 0; i < length; i++) left.grad[i] += scalar * out.grad[i];
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
    assert(shape[1] == other.shape[0],
        "Dimension mismatch: ${shape[1]} != ${other.shape[0]}");

    // CRITICAL: Capture 'this' and 'other' as local final variables.
    // This ensures the backward closure points to the EXACT weight buffers.
    final Tensor input = this;
    final Tensor weights = other;

    int M = shape[0];
    int K = shape[1];
    int N = other.shape[1];

    final out = Tensor([M, N], children: {input, weights});

    // Forward Pass
    for (int i = 0; i < M; i++) {
      for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
          out.data[i * N + j] +=
              input.data[i * K + k] * weights.data[k * N + j];
        }
      }
    }

    // Backward Pass
    out.onBackward = () {
      for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
          for (int j = 0; j < N; j++) {
            double gradOut = out.grad[i * N + j];
            // Update the locally captured weight and input tensors
            input.grad[i * K + k] += weights.data[k * N + j] * gradOut;
            weights.grad[k * N + j] += input.data[i * K + k] * gradOut;
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
      final Tensor self = this;
      for (int i = 0; i < length; i++) {
        double x = self.data[i];
        // Numerical stability check
        if (x.isNaN || x.isInfinite) continue;

        double v = 0.7978845608 * (x + 0.044715 * math.pow(x, 3));
        double t = _mathTanh(v);
        double deriv = 0.5 * (1.0 + t) +
            0.5 *
                x *
                (1.0 - t * t) *
                0.7978845608 *
                (1.0 + 3.0 * 0.044715 * x * x);

        double incomingGrad = out.grad[i];
        if (!incomingGrad.isNaN) {
          self.grad[i] += deriv * incomingGrad;
        }
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
    for (int i = 0; i < length; i++) {
      out.data[i] /= (sumExp + 1e-9);
    }
    out.onBackward = () {
      double dot = 0;
      for (int i = 0; i < length; i++) {
        dot += out.data[i] * out.grad[i];
      }
      for (int i = 0; i < length; i++) {
        grad[i] += out.data[i] * (out.grad[i] - dot);
      }
    };
    return out;
  }

  Tensor crossEntropy(Tensor target) {
    final out = Tensor([1], children: {this});
    double lossSum = 0;
    // Assumes 'this' is already passed through Softmax
    for (int i = 0; i < length; i++) {
      lossSum -= target.data[i] * math.log(data[i] + 1e-9);
    }
    out.data[0] = lossSum / (shape[0]); // Average over batch

    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        // Gradient of Softmax + CrossEntropy simplifies to (pred - target)
        grad[i] += (data[i] - target.data[i]) * out.grad[0];
      }
    };
    return out;
  }

  Tensor transpose() {
    final out = Tensor([shape[1], shape[0]], children: {this});
    int R = shape[0];
    int C = shape[1];
    for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
        out.data[j * R + i] = data[i * C + j];
      }
    }
    out.onBackward = () {
      for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
          grad[i * C + j] += out.grad[j * R + i];
        }
      }
    };
    return out;
  }

  static Tensor concat(List<Tensor> tensors, {int axis = 1}) {
    if (tensors.isEmpty) throw ArgumentError("Cannot concat empty list");
    if (axis != 1)
      throw UnimplementedError("Only axis 1 concat is implemented for now");

    // 1. Calculate the new shape
    int rows = tensors[0].shape[0];
    int newCols = 0;
    for (var t in tensors) {
      if (t.shape[0] != rows)
        throw ArgumentError("All tensors must have same number of rows");
      newCols += t.shape[1];
    }

    final out = Tensor([rows, newCols], children: tensors);

    // 2. Forward Pass: Copy data into the new buffer
    int currentOffset = 0;
    for (var t in tensors) {
      int tWidth = t.shape[1];
      for (int i = 0; i < rows; i++) {
        // Copy one row of the current tensor into the correct spot in 'out'
        for (int j = 0; j < tWidth; j++) {
          out.data[i * newCols + (currentOffset + j)] = t.data[i * tWidth + j];
        }
      }
      currentOffset += tWidth;
    }

    // 3. Backward Pass: Map gradients back to the original tensors
    out.onBackward = () {
      int backOffset = 0;
      for (var t in tensors) {
        int tWidth = t.shape[1];
        for (int i = 0; i < rows; i++) {
          for (int j = 0; j < tWidth; j++) {
            t.grad[i * tWidth + j] += out.grad[i * newCols + (backOffset + j)];
          }
        }
        backOffset += tWidth;
      }
    };

    return out;
  }

  // --- Slicing ---

  Tensor slice(int startOffset, int endOffset) {
    final int sliceLength = endOffset - startOffset;
    final out = Tensor([sliceLength], children: {this});
    for (int i = 0; i < sliceLength; i++) {
      out.data[i] = data[startOffset + i];
    }
    out.onBackward = () {
      for (int i = 0; i < sliceLength; i++) {
        grad[startOffset + i] += out.grad[i];
      }
    };
    return out;
  }

  /// Extracts a specific row as a new [1, Cols] Tensor
  Tensor getRow(int rowIndex) {
    final int cols = shape[1];
    final int start = rowIndex * cols;
    final out = Tensor([1, cols], children: {this});

    for (int i = 0; i < cols; i++) {
      out.data[i] = data[start + i];
    }

    out.onBackward = () {
      for (int i = 0; i < cols; i++) {
        grad[start + i] += out.grad[i];
      }
    };
    return out;
  }

  /// In-place copy of row data (Used for assembling the final Y matrix)
  void setRow(int rowIndex, Tensor rowTensor) {
    final int cols = shape[1];
    final int start = rowIndex * cols;
    for (int i = 0; i < cols; i++) {
      data[start + i] = rowTensor.data[i];
    }
    // Note: setRow is used at the end of forward, gradients are
    // handled by the graph of the Tensors created during calculation.
  }

  // Tensor slice2D(int r, int c) {
  //   final out = Tensor([r, c], children: {this});
  //   for (int i = 0; i < r; i++) {
  //     for (int j = 0; j < c; j++) {
  //       out.data[i * c + j] = data[i * shape[1] + j];
  //     }
  //   }
  //   out.onBackward = () {
  //     for (int i = 0; i < r; i++) {
  //       for (int j = 0; j < c; j++) {
  //         grad[i * shape[1] + j] += out.grad[i * c + j];
  //       }
  //     }
  //   };
  //   return out;
  // }

  Tensor slice2D(int rows, int cols) {
    // 1. Safety Check: Prevent RangeErrors before they happen
    if (rows > this.shape[0] || cols > this.shape[1]) {
      throw RangeError(
          "Slice dimensions [$rows, $cols] exceed Tensor shape ${this.shape}");
    }

    final out = Tensor([rows, cols], children: {this});
    final int stride = this.shape[1]; // The original width

    // 2. Forward Pass
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        out.data[i * cols + j] = this.data[i * stride + j];
      }
    }

    // 3. Backward Pass: Map gradients from the small slice back to the big tensor
    out.onBackward = () {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          // We use += because multiple slices might overlap or be used multiple times
          this.grad[i * stride + j] += out.grad[i * cols + j];
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

  Tensor leakyRelu([double alpha = 0.01]) {
    final out = Tensor(shape, children: {this});
    for (int i = 0; i < length; i++) {
      out.data[i] = data[i] > 0 ? data[i] : data[i] * alpha;
    }
    out.onBackward = () {
      for (int i = 0; i < length; i++) {
        grad[i] += (data[i] > 0 ? 1.0 : alpha) * out.grad[i];
      }
    };
    return out;
  }

  // 2. Fix the setter to ensure it's hitting the private field
  set onBackward(void Function() func) {
    _backward = func;
  }

  void _runBackward() {
    if (_backward != null) _backward!();
  }

  Set<Tensor> get parents => _prev;

  // Inside your Tensor class
  // 3. Update the loop in backward()
  void backward({bool seed = true}) {
    final topo = <Tensor>[];
    final visited = <Tensor>{};
    void build(Tensor t) {
      if (visited.add(t)) {
        for (final p in t.parents) build(p);
        topo.add(t);
      }
    }

    build(this);

    if (seed) grad.fillRange(0, length, 1.0);

    int ran = 0;
    for (final t in topo.reversed) {
      // Direct call to our helper
      t._runBackward();
      ran++;
    }
    // print("Backprop: $ran functions executed.");
  }

  void zeroGrad() => grad.fillRange(0, length, 0.0);

  // Add this helper to your Tensor or Generation logic
// List<double> softmax(List<double> logits) {
//   double maxLogit = logits.reduce(math.max);
//   List<double> exps = logits.map((l) => math.exp(l - maxLogit)).toList();
//   double sumExps = exps.reduce((a, b) => a + b);
//   return exps.map((e) => e / sumExps).toList();
// }
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
