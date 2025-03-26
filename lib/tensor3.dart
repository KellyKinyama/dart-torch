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
    final reshapedData = _reshapeTo3D(flattened, newShape);

    return Tensor(reshapedData);
  }

  // Convert the flattened data back into the 3D structure
  List<List<List<double>>> _reshapeTo3D(
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

  // Element-wise addition
  Tensor add(Tensor other) {
    if (shape[0] != other.shape[0] ||
        shape[1] != other.shape[1] ||
        shape[2] != other.shape[2]) {
      throw ArgumentError(
          'Tensors must have the same shape for element-wise addition');
    }

    final result = List.generate(
        shape[0],
        (i) => List.generate(
            shape[1],
            (j) => List.generate(
                shape[2], (k) => data[i][j][k] + other.data[i][j][k])));

    return Tensor(result);
  }

  // Element-wise multiplication
  Tensor multiply(Tensor other) {
    if (shape[0] != other.shape[0] ||
        shape[1] != other.shape[1] ||
        shape[2] != other.shape[2]) {
      throw ArgumentError(
          'Tensors must have the same shape for element-wise multiplication');
    }

    final result = List.generate(
        shape[0],
        (i) => List.generate(
            shape[1],
            (j) => List.generate(
                shape[2], (k) => data[i][j][k] * other.data[i][j][k])));

    return Tensor(result);
  }

  // Scalar multiplication
  Tensor scalarMultiply(double scalar) {
    final result = List.generate(
        shape[0],
        (i) => List.generate(shape[1],
            (j) => List.generate(shape[2], (k) => data[i][j][k] * scalar)));

    return Tensor(result);
  }

  @override
  String toString() {
    return 'Tensor(shape: $shape, data: $data)';
  }
}

void main() {
  // Example of tensor creation and reshaping
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

  // Reshape the tensor to 2x3x2
  final reshapedTensor = tensor1.reshape([2, 3, 2]);
  print('Reshaped Tensor: $reshapedTensor');
}
