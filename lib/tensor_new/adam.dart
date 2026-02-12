import 'dart:math' as math;
import '../tensor/tensor.dart';

class Adam {
  final List<Tensor> parameters;
  final double lr;
  final double beta1;
  final double beta2;
  final double epsilon;

  // State buffers for moments
  final List<List<double>> m; // 1st moment (mean)
  final List<List<double>> v; // 2nd moment (uncentered variance)
  int t = 0; // Timestep for bias correction

  Adam(this.parameters,
      {this.lr = 0.001,
      this.beta1 = 0.9,
      this.beta2 = 0.999,
      this.epsilon = 1e-8})
      : m = List.generate(parameters.length,
            (i) => List.filled(parameters[i].data.length, 0.0)),
        v = List.generate(parameters.length,
            (i) => List.filled(parameters[i].data.length, 0.0));

  void step() {
    t++;
    // Bias correction coefficients
    final double bc1 = 1.0 - math.pow(beta1, t);
    final double bc2 = 1.0 - math.pow(beta2, t);

    for (int i = 0; i < parameters.length; i++) {
      final p = parameters[i];
      for (int j = 0; j < p.data.length; j++) {
        final g = p.grad[j];

        // Update biased first moment estimate
        m[i][j] = beta1 * m[i][j] + (1.0 - beta1) * g;
        // Update biased second raw moment estimate
        v[i][j] = beta2 * v[i][j] + (1.0 - beta2) * (g * g);

        // Compute bias-corrected estimates
        final mHat = m[i][j] / bc1;
        final vHat = v[i][j] / bc2;

        // Update parameters
        p.data[j] -= lr * mHat / (math.sqrt(vHat) + epsilon);
      }
    }
  }

  void zeroGrad() {
    for (var p in parameters) {
      p.grad.fillRange(0, p.grad.length, 0.0);
    }
  }
}
