import 'dart:math';

class Tensor {
  List<List<List<double>>> data;
  final List<int> shape;
  Tensor? grad; // Gradient (for backpropagation)
  final Function? _backward; // Backward function to compute the gradient

  Tensor(this.data, {this.grad, Function? backward})
      : shape = _getShape(data),
        _backward = backward;

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

    return Tensor(reshapedData, backward: _backward);
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

    return Tensor(result, backward: () {
      grad?.data =
          grad?.data.map((e) => e.map((e1) => e1 + other.grad!.data)).toList();
    });
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

    return Tensor(result, backward: () {
      grad?.data =
          grad?.data.map((e) => e.map((e1) => e1 * other.grad!.data)).toList();
    });
  }

  // Scalar multiplication
  Tensor scalarMultiply(double scalar) {
    final result = List.generate(
        shape[0],
        (i) => List.generate(shape[1],
            (j) => List.generate(shape[2], (k) => data[i][j][k] * scalar)));

    return Tensor(result, backward: () {
      grad?.data = grad?.data.map((e) => e.map((e1) => e1 * scalar)).toList();
    });
  }

  // Initialize backward function for reverse mode gradients
  void backward() {
    if (_backward != null) {
      _backward!();
    }
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

  final tensor2 = Tensor([
    [
      [0.1, 0.2, 0.3],
      [0.4, 0.5, 0.6],
    ],
    [
      [0.7, 0.8, 0.9],
      [1.0, 1.1, 1.2],
    ]
  ]);

  final sum = tensor1.add(tensor2);
  sum.backward(); // Compute gradient

  final product = tensor1.multiply(tensor2);
  product.backward(); // Compute gradient

  final scalar = tensor1.scalarMultiply(2);
  scalar.backward(); // Compute gradient

  print('Sum Tensor: $sum');
  print('Product Tensor: $product');
  print('Scalar Multiplied Tensor: $scalar');
}
