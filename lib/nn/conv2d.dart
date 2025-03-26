import 'dart:math' as math;
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
import 'matrix2d.dart';

class Conv2d {
  final int inChannels;
  final int outChannels;
  final int kernelSize;
  final int stride;
  final int padding;
  final bool useBias;
  late List<Matrix2d> kernels;
  late ValueVector biases;

  Conv2d({
    required this.inChannels,
    required this.outChannels,
    required this.kernelSize,
    this.stride = 1,
    this.padding = 0,
    this.useBias = true,
  }) {
    // Initialize filters (kernels) with small random values
    kernels = List.generate(outChannels, (i) {
      return Matrix2d(
          kernelSize,
          kernelSize,
          ValueVector(List.generate(kernelSize * kernelSize,
              (j) => Value(math.Random().nextDouble() * 0.1 - 0.05))));
    });

    // Initialize biases (if used)
    biases = ValueVector(List.generate(outChannels, (i) => Value(0)));
  }

  Matrix2d forward(Matrix2d input) {
    if (padding > 0) {
      input = input.pad(padding);
    }

    int outputRows = ((input.rows() - kernelSize) ~/ stride) + 1;
    int outputCols = ((input.cols() - kernelSize) ~/ stride) + 1;
    Matrix2d output = Matrix2d(outputRows, outputCols);

    for (int filterIdx = 0; filterIdx < outChannels; filterIdx++) {
      Matrix2d kernel = kernels[filterIdx];
      Value bias = biases.values[filterIdx];

      for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
          Value sum = Value(0);
          for (int ki = 0; ki < kernelSize; ki++) {
            for (int kj = 0; kj < kernelSize; kj++) {
              int row = i * stride + ki;
              int col = j * stride + kj;
              sum += input.at(row, col) * kernel.at(ki, kj);
            }
          }
          sum += bias;
          output.data!.values[i * outputCols + j] = sum;
        }
      }
    }

    return output;
  }
}

void main() {
  Matrix2d input = Matrix2d(5, 5); // Example 5x5 input matrix
  Conv2d convLayer = Conv2d(
      inChannels: 1, outChannels: 2, kernelSize: 3, stride: 1, padding: 1);

  Matrix2d output = convLayer.forward(input);
  print("Output Matrix: ${output.data!.values}");
}
