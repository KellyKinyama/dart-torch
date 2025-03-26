import 'dart:math';

class Tensor {
  final List<List<List<double>>> data;
  final List<int> shape;

  Tensor(this.data) : shape = _getShape(data);

  // Helper function to get the shape of the tensor
  static List<int> _getShape(List<List<List<double>>> data) {
    return [data.length, data[0].length, data[0][0].length];
  }

  // Helper function to flatten the tensor data into a single list
  List<double> _flatten() {
    return data.expand((x) => x.expand((y) => y)).toList();
  }

  // Reshape the tensor into the new shape
  Tensor reshape(List<int> newShape) {
    int totalElements = shape.fold(1, (a, b) => a * b);
    int newTotalElements = newShape.fold(1, (a, b) => a * b);

    if (totalElements != newTotalElements) {
      throw ArgumentError(
          'The total number of elements must remain the same when reshaping');
    }

    final flattened = _flatten();
    final reshapedData = _reshapeToNDim(flattened, newShape);

    return Tensor(reshapedData);
  }

  // Convert the flattened data back into the N-dimensional structure
  List<List<List<double>>> _reshapeToNDim(
      List<double> flattened, List<int> newShape) {
    int depth = newShape[0];
    int rows = newShape[1];
    int cols = newShape[2];

    List<List<List<double>>> reshapedData = List.generate(
        depth,
        (i) => List.generate(
            rows, (j) => List.generate(cols, (k) => flattened.removeAt(0))));

    return reshapedData;
  }

  @override
  String toString() {
    return 'Tensor(shape: $shape, data: $data)';
  }
}

void main() {
  final tensor1 = Tensor([
    [
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ],
    [
      [7.0, 8.0, 9.0],
      [10.0, 11.0, 12.0],
    ]
  ]);

  print('Original Tensor: $tensor1');

  // Example 1: Reshape 2x2x3 Tensor to 3x2x2
  final reshapedTensor1 = tensor1.reshape([3, 2, 2]);
  print('Reshaped Tensor (3x2x2): $reshapedTensor1');

  // Example 2: Reshape 2x2x3 Tensor to 6x2
  final reshapedTensor2 = tensor1.reshape([6, 2]);
  print('Reshaped Tensor (6x2): $reshapedTensor2');

  // Example 3: Reshape 2x2x3 Tensor to 4x3
  final reshapedTensor3 = tensor1.reshape([4, 3]);
  print('Reshaped Tensor (4x3): $reshapedTensor3');

  // Example 4: Reshape 2x2x3 Tensor to 1x12
  final reshapedTensor4 = tensor1.reshape([1, 12]);
  print('Reshaped Tensor (1x12): $reshapedTensor4');

  // Example 5: Reshape 2x2x3 Tensor to 3x4 (invalid)
  try {
    final reshapedTensor5 = tensor1.reshape([3, 4]);
    print('Reshaped Tensor (3x4): $reshapedTensor5');
  } catch (e) {
    print('Error: $e');
  }

  // Example 6: Reshape 2x2x3 Tensor to 4x3x1
  final reshapedTensor6 = tensor1.reshape([4, 3, 1]);
  print('Reshaped Tensor (4x3x1): $reshapedTensor6');

  // Example 7: Reshape 2x2x3 Tensor to 2x6
  final reshapedTensor7 = tensor1.reshape([2, 6]);
  print('Reshaped Tensor (2x6): $reshapedTensor7');

  // Example 8: Reshape in steps (3x2x2 to 6x2)
  final reshapedTensor1a = tensor1.reshape([3, 2, 2]);
  print('Step 1 Reshaped Tensor (3x2x2): $reshapedTensor1a');

  final reshapedTensor8 = reshapedTensor1a.reshape([6, 2]);
  print('Step 2 Reshaped Tensor (6x2): $reshapedTensor8');
}
