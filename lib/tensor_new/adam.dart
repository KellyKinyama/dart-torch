import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';

class Adam {
  final List<Tensor> params;
  double lr, beta1, beta2, eps;
  int t = 0;
  late List<Float32List> m, v;

  Adam(this.params,
      {this.lr = 0.001,
      this.beta1 = 0.9,
      this.beta2 = 0.999,
      this.eps = 1e-8}) {
    m = params.map((p) => Float32List(p.length)).toList();
    v = params.map((p) => Float32List(p.length)).toList();
  }

  void step() {
    t++;
    for (int i = 0; i < params.length; i++) {
      final p = params[i];
      for (int j = 0; j < p.length; j++) {
        m[i][j] = beta1 * m[i][j] + (1 - beta1) * p.grad[j];
        v[i][j] = beta2 * v[i][j] + (1 - beta2) * p.grad[j] * p.grad[j];

        double mHat = m[i][j] / (1 - math.pow(beta1, t));
        double vHat = v[i][j] / (1 - math.pow(beta2, t));

        p.data[j] -= lr * mHat / (math.sqrt(vHat) + eps);
      }
    }
  }
}
