import 'dart:math' as math;
import 'dart:typed_data';

import '../tensor/tensor.dart';
import 'module.dart';

class LayerNorm extends Module {
  final Tensor gamma; // Scale
  final Tensor beta; // Shift
  final double eps;

  LayerNorm(int dim, {this.eps = 1e-5})
      : gamma = Tensor.fill([1, dim], 1.0),
        beta = Tensor.fill([1, dim], 0.0);

  Tensor forward(Tensor x) {
    // x shape: [Rows, Cols] (e.g., [seqLength, embedSize])
    int R = x.shape[0];
    int C = x.shape[1];
    final out = Tensor(x.shape, children: {x, gamma, beta});

    // We'll need these for the backward pass
    final means = Float32List(R);
    final vars = Float32List(R);

    // 1. Forward Pass
    for (int i = 0; i < R; i++) {
      double sum = 0;
      for (int j = 0; j < C; j++) sum += x.data[i * C + j];
      double mean = sum / C;
      means[i] = mean;

      double sqDiffSum = 0;
      for (int j = 0; j < C; j++) {
        double diff = x.data[i * C + j] - mean;
        sqDiffSum += diff * diff;
      }
      double variance = sqDiffSum / C;
      vars[i] = variance;

      double stdInv = 1.0 / math.sqrt(variance + eps);

      for (int j = 0; j < C; j++) {
        double xHat = (x.data[i * C + j] - mean) * stdInv;
        out.data[i * C + j] = xHat * gamma.data[j] + beta.data[j];
      }
    }

    // 2. Backward Pass
    out.onBackward = () {
      for (int i = 0; i < R; i++) {
        double stdInv = 1.0 / math.sqrt(vars[i] + eps);

        // Sums needed for the derivative of the mean and variance
        double dlDxhatSum = 0;
        double dlDxhatXhatSum = 0;

        for (int j = 0; j < C; j++) {
          double xHat = (x.data[i * C + j] - means[i]) * stdInv;
          double dlDxhat = out.grad[i * C + j] * gamma.data[j];

          dlDxhatSum += dlDxhat;
          dlDxhatXhatSum += dlDxhat * xHat;

          // Gradients for learnable params gamma and beta
          gamma.grad[j] += out.grad[i * C + j] * xHat;
          beta.grad[j] += out.grad[i * C + j];
        }

        // The gradient of the input x (applying the chain rule for LayerNorm)
        for (int j = 0; j < C; j++) {
          double xHat = (x.data[i * C + j] - means[i]) * stdInv;
          double dlDxhat = out.grad[i * C + j] * gamma.data[j];

          x.grad[i * C + j] +=
              (stdInv / C) * (C * dlDxhat - dlDxhatSum - xHat * dlDxhatXhatSum);
        }
      }
    };

    return out;
  }

  @override
  List<Tensor> parameters() => [gamma, beta];
}
