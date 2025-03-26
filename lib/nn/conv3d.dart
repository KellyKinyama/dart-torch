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
  late List<List<Matrix2d>>
      kernels; // 3D list: [outChannels][inChannels][Kernel Matrix]
  late ValueVector biases;

  Conv2d({
    required this.inChannels,
    required this.outChannels,
    required this.kernelSize,
    this.stride = 1,
    this.padding = 0,
    this.useBias = true,
  }) {
    // Initialize kernels for each output channel with weights for each input channel
    kernels = List.generate(outChannels, (outIdx) {
      return List.generate(inChannels, (inIdx) {
        return Matrix2d(
          kernelSize,
          kernelSize,
          ValueVector(List.generate(kernelSize * kernelSize,
              (i) => Value(math.Random().nextDouble() * 0.1 - 0.05))),
        );
      });
    });

    // Initialize biases (if used)
    biases = ValueVector(List.generate(outChannels, (i) => Value(0)));
  }

  /// Applies 2D convolution to multi-channel input
  List<Matrix2d> forward(List<Matrix2d> input) {
    if (input.length != inChannels) {
      throw ArgumentError(
          "Expected $inChannels input channels, got ${input.length}");
    }

    // Apply padding to each input channel if needed
    List<Matrix2d> paddedInputs =
        input.map((mat) => padding > 0 ? mat.pad(padding) : mat).toList();

    int outputRows = ((paddedInputs[0].rows() - kernelSize) ~/ stride) + 1;
    int outputCols = ((paddedInputs[0].cols() - kernelSize) ~/ stride) + 1;

    // Output feature maps for each output channel
    List<Matrix2d> outputFeatureMaps =
        List.generate(outChannels, (_) => Matrix2d(outputRows, outputCols));

    for (int outIdx = 0; outIdx < outChannels; outIdx++) {
      Matrix2d output = outputFeatureMaps[outIdx];
      Value bias = biases.values[outIdx];

      for (int i = 0; i < outputRows; i++) {
        for (int j = 0; j < outputCols; j++) {
          Value sum = Value(0);

          // Sum over all input channels
          for (int inIdx = 0; inIdx < inChannels; inIdx++) {
            Matrix2d inputChannel = paddedInputs[inIdx];
            Matrix2d kernel = kernels[outIdx][inIdx];

            for (int ki = 0; ki < kernelSize; ki++) {
              for (int kj = 0; kj < kernelSize; kj++) {
                int row = i * stride + ki;
                int col = j * stride + kj;
                sum += inputChannel.at(row, col) * kernel.at(ki, kj);
              }
            }
          }

          // Add bias and store result
          sum += bias;
          output.data!.values[i * outputCols + j] = sum;
        }
      }
    }

    return outputFeatureMaps;
  }
}

void main() {
  // Example 3-channel input (like an RGB image with 5x5 pixels)
  List<Matrix2d> input = [
    Matrix2d(5, 5), // Red channel
    Matrix2d(5, 5), // Green channel
    Matrix2d(5, 5), // Blue channel
  ];

  // Conv layer: 3 input channels, 2 output channels, 3x3 kernel, stride 1, padding 1
  Conv2d convLayer = Conv2d(
      inChannels: 3, outChannels: 2, kernelSize: 3, stride: 1, padding: 1);

  // Forward pass
  List<Matrix2d> output = convLayer.forward(input);

  // Print output dimensions
  print("Output Feature Map Sizes: ${output[0].rows()}x${output[0].cols()}");
}
