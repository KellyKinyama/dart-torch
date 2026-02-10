import 'dart:math' as math;
import 'dart:typed_data';

import 'module.dart';
import 'tensor.dart';

class LayerNorm extends Module {
  final Tensor gamma;
  final Tensor beta;
  final double eps = 1e-5;

  LayerNorm(int dim)
      : gamma = Tensor.fill([1, dim], 1.0),
        beta = Tensor.fill([1, dim], 0.0);

  @override
  Tensor forward(Tensor x) {
    final out = Tensor(x.shape, children: {x, gamma, beta});
    final int rows = x.shape[0];
    final int cols = x.shape[1];

    // Buffers to store intermediate values for the backward pass
    final means = Float32List(rows);
    final stds = Float32List(rows);

    // Forward Pass
    for (int i = 0; i < rows; i++) {
      double sum = 0;
      for (int j = 0; j < cols; j++) sum += x.data[i * cols + j];
      double mean = sum / cols;
      means[i] = mean;

      double varSum = 0;
      for (int j = 0; j < cols; j++) {
        double diff = x.data[i * cols + j] - mean;
        varSum += diff * diff;
      }
      double std = math.sqrt(varSum / cols + eps);
      stds[i] = std;

      for (int j = 0; j < cols; j++) {
        int idx = i * cols + j;
        out.data[idx] =
            ((x.data[idx] - mean) / std) * gamma.data[j] + beta.data[j];
      }
    }

    // Use the setter for the LayerNorm backward logic
    out.onBackward = () {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          int idx = i * cols + j;
          double normalized = (x.data[idx] - means[i]) / stds[i];

          // 1. Gradients for Gamma and Beta
          gamma.grad[j] += out.grad[idx] * normalized;
          beta.grad[j] += out.grad[idx];

          // 2. Gradient for Input (x)
          // Simplified version: pass the gradient back through the normalization
          double dNorm = out.grad[idx] * gamma.data[j];
          x.grad[idx] += (1.0 / (cols * stds[i])) *
              (cols * dNorm -
                  sumGradients(dNorm) -
                  normalized * sumGradients(dNorm * normalized));
        }
      }
    };

    return out;
  }

  // Helper for LayerNorm backward logic
  double sumGradients(double val) => val; // Conceptual placeholder

  @override
  List<Tensor> parameters() => [gamma, beta];
}
