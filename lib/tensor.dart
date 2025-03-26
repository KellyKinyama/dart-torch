import 'dart:math';

class Tensor {
  final List<List<List<double>>> data;
  final List<int> shape;

  Tensor(this.data) : shape = _getShape(data);

  // Helper function to get the shape of the tensor
  static List<int> _getShape(List<List<List<double>>> data) {
    return [data.length, data[0].length, data[0][0].length];
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
  // Example of tensor creation and operations
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

  final tensor2 = Tensor([
    [
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
    ],
    [
      [1.0, 1.0, 1.0],
      [1.0, 1.0, 1.0],
    ]
  ]);

  final result = tensor1.add(tensor2);
  print('Addition Result: $result');

  final multiplied = tensor1.multiply(tensor2);
  print('Multiplication Result: $multiplied');
}
