import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';
import 'matrix2d.dart';

enum PoolingType { max, average }

class Pooling2d {
  final int kernelSize;
  final int stride;
  final PoolingType type;

  Pooling2d(
      {required this.kernelSize, required this.stride, required this.type});

  /// Applies pooling to a **single-channel matrix**
  Matrix2d _poolSingle(Matrix2d input) {
    int outputRows = ((input.rows() - kernelSize) ~/ stride) + 1;
    int outputCols = ((input.cols() - kernelSize) ~/ stride) + 1;

    Matrix2d output = Matrix2d(outputRows, outputCols);

    for (int i = 0; i < outputRows; i++) {
      for (int j = 0; j < outputCols; j++) {
        List<Value> window = [];

        // Extract pooling window
        for (int ki = 0; ki < kernelSize; ki++) {
          for (int kj = 0; kj < kernelSize; kj++) {
            int row = i * stride + ki;
            int col = j * stride + kj;
            window.add(input.at(row, col));
          }
        }

        // Apply pooling operation
        if (type == PoolingType.max) {
          output.data!.values[i * outputCols + j] =
              window.reduce((a, b) => a.data > b.data ? a : b);
        } else {
          Value sum = window.reduce((a, b) => a + b);
          output.data!.values[i * outputCols + j] = sum / window.length;
        }
      }
    }

    return output;
  }

  /// Applies pooling to **multi-channel inputs** (list of `Matrix2d`)
  List<Matrix2d> forward(List<Matrix2d> input) {
    return input.map(_poolSingle).toList();
  }
}

void main() {
  // Example 3-channel input (like an RGB image with 5x5 pixels)
  List<Matrix2d> input = [
    Matrix2d(5, 5), // Red channel
    Matrix2d(5, 5), // Green channel
    Matrix2d(5, 5), // Blue channel
  ];

  // Create Max Pooling Layer (2x2 kernel, stride 2)
  Pooling2d maxPool =
      Pooling2d(kernelSize: 2, stride: 2, type: PoolingType.max);
  List<Matrix2d> maxPooledOutput = maxPool.forward(input);
  print(
      "Max Pooling Output Size: ${maxPooledOutput[0].rows()}x${maxPooledOutput[0].cols()}");

  // Create Average Pooling Layer (2x2 kernel, stride 2)
  Pooling2d avgPool =
      Pooling2d(kernelSize: 2, stride: 2, type: PoolingType.average);
  List<Matrix2d> avgPooledOutput = avgPool.forward(input);
  print(
      "Average Pooling Output Size: ${avgPooledOutput[0].rows()}x${avgPooledOutput[0].cols()}");
}
