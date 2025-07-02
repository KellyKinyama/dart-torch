// value_matrix.dart (new file)
//import 'package:vector_math/vector_math.dart'; // Consider using a math library for matrix ops, or implement from scratch
import '/nn/value.dart';

class ValueMatrix {
  final List<List<Value>> _data;
  final int rows;
  final int cols;

  ValueMatrix(this._data)
      : rows = _data.length,
        cols = _data.isEmpty ? 0 : _data[0].length {
    if (_data.any((row) => row.length != cols)) {
      throw ArgumentError("All rows must have the same number of columns.");
    }
  }

  // Factory constructor for creating a matrix of specified dimensions with initial values
  factory ValueMatrix.zeros(int rows, int cols) {
    return ValueMatrix(
        List.generate(rows, (_) => List.generate(cols, (_) => Value(0.0))));
  }

  // Get a Value at specific row and column
  Value at(int row, int col) => _data[row][col];

  // Matrix multiplication (A * B)
  ValueMatrix multiply(ValueMatrix other) {
    if (cols != other.rows) {
      throw ArgumentError(
          "Number of columns in first matrix must match number of rows in second matrix.");
    }

    final result = List<List<Value>>.generate(rows, (r) {
      return List<Value>.generate(other.cols, (c) {
        Value sum = Value(0.0);
        for (int i = 0; i < cols; i++) {
          sum += _data[r][i] * other._data[i][c];
        }
        return sum;
      });
    });
    return ValueMatrix(result);
  }

  // Transpose of the matrix
  ValueMatrix transpose() {
    final result = List<List<Value>>.generate(cols, (c) {
      return List<Value>.generate(rows, (r) => _data[r][c]);
    });
    return ValueMatrix(result);
  }

  // Scalar addition
  ValueMatrix operator +(dynamic other) {
    if (other is Value) {
      return ValueMatrix(
          _data.map((row) => row.map((v) => v + other).toList()).toList());
    } else if (other is ValueMatrix) {
      if (rows != other.rows || cols != other.cols) {
        throw ArgumentError(
            "Matrices must have the same dimensions for addition.");
      }
      return ValueMatrix(List.generate(rows,
          (r) => List.generate(cols, (c) => _data[r][c] + other._data[r][c])));
    }
    throw UnimplementedError(
        "Addition not supported for type ${other.runtimeType}");
  }

  // Scalar multiplication
  ValueMatrix operator *(dynamic other) {
    if (other is Value) {
      return ValueMatrix(
          _data.map((row) => row.map((v) => v * other).toList()).toList());
    }
    throw UnimplementedError(
        "Multiplication not supported for type ${other.runtimeType}");
  }

  // Element-wise operations (example: applying activation function)
  ValueMatrix sigmoid() {
    return ValueMatrix(
        _data.map((row) => row.map((v) => v.sigmoid()).toList()).toList());
  }

  ValueMatrix relu() {
    return ValueMatrix(
        _data.map((row) => row.map((v) => v.relu()).toList()).toList());
  }

  ValueMatrix tanh() {
    return ValueMatrix(
        _data.map((row) => row.map((v) => v.tanh()).toList()).toList());
  }

  // Flatten to a single list of Values (useful for parameters)
  List<Value> flatten() {
    return _data.expand((row) => row).toList();
  }

  @override
  String toString() {
    return _data
        .map((row) => row.map((v) => v.data.toStringAsFixed(2)).join('\t'))
        .join('\n');
  }
}
