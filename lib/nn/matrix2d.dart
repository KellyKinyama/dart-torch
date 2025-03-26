import 'dart:math' as math;
import 'package:dart_torch/nn/value.dart';
import 'package:dart_torch/nn/value_vector.dart';

class Matrix2d {
  ValueVector? data;

  int _cols;
  int _rows;

  Matrix2d(this._rows, this._cols, [this.data]) {
    data ??
        (data = ValueVector(List<Value>.generate(
            _cols * _rows, (int index) => Value(math.Random().nextDouble()),
            growable: false)));
  }

  int rows() {
    return _rows;
  }

  int cols() {
    return _cols;
  }

  Value at(int row, int col) {
    return data!.values[row * _cols + col];
  }

  Matrix2d operator +(dynamic other) {
    if (other is num) {
      return Matrix2d(
          _rows,
          _cols,
          ValueVector(List.generate(_cols * _rows, (int index) {
            return data!.values[index] + other;
          })));
    } else {
      if (other is Matrix2d) {
        return Matrix2d(
            _rows,
            _cols,
            ValueVector(List.generate(_cols * _rows, (int index) {
              return data!.values[index] + other.data!.values[index];
            })));
      } else {
        throw UnimplementedError(
            "Operation + not supported for: ${other.runtimeType}");
      }
    }
  }

  Matrix2d operator -(dynamic other) {
    if (other is num) {
      return Matrix2d(
          _rows,
          _cols,
          ValueVector(List.generate(_cols * _rows, (int index) {
            return data!.values[index] - other;
          })));
    } else {
      if (other is Matrix2d) {
        return Matrix2d(
            _rows,
            _cols,
            ValueVector(List.generate(_cols * _rows, (int index) {
              return data!.values[index] - other.data!.values[index];
            })));
      } else {
        throw UnimplementedError(
            "Operation + not supported for: ${other.runtimeType}");
      }
    }
  }

  Matrix2d operator /(dynamic other) {
    if (other is num) {
      return Matrix2d(
          _rows,
          _cols,
          ValueVector(List.generate(_cols * _rows, (int index) {
            return data!.values[index] / other;
          })));
    } else {
      if (other is Matrix2d) {
        // return Matrix2d(
        //     _rows,
        //     _cols,
        //     ValueVector(List.generate(_cols * _rows, (int index) {
        //       return data!.values[index] - other.data!.values[index];
        //     })));
        throw UnimplementedError(
            "Operation / not supported for: ${other.runtimeType}");
      } else {
        throw UnimplementedError(
            "Operation + not supported for: ${other.runtimeType}");
      }
    }
  }

  Matrix2d operator *(dynamic other) {
    if (other is num) {
      return Matrix2d(
          _rows,
          _cols,
          ValueVector(List.generate(_cols * _rows, (int index) {
            return data!.values[index] * other;
          })));
    } else if (other is Matrix2d) {
      if (_cols != other._rows) {
        throw ArgumentError("Matrix multiplication requires A.cols == B.rows.");
      }
      Matrix2d result = Matrix2d(_rows, other._cols);
      for (int i = 0; i < _rows; i++) {
        for (int j = 0; j < other._cols; j++) {
          Value sum = Value(0);
          for (int k = 0; k < _cols; k++) {
            sum += at(i, k) * other.at(k, j);
          }
          result.data!.values[(i * other._cols + j).toInt()] = sum;
        }
      }
      return result;
    } else {
      throw UnimplementedError(
          "Operation * not supported for: ${other.runtimeType}");
    }
  }

  Matrix2d pad(int rings) {
    int newRows = _rows + 2 * rings;
    int newCols = _cols + 2 * rings;
    Matrix2d padded = Matrix2d(
        newRows,
        newCols,
        ValueVector(List.generate(newRows * newCols, (index) => Value(0),
            growable: false)));

    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        padded.data!.values[(i + rings) * newCols + (j + rings)] = at(i, j);
      }
    }
    return padded;
  }

  Matrix2d transpose() {
    Matrix2d result = Matrix2d(_cols, _rows);
    for (int i = 0; i < _rows; i++) {
      for (int j = 0; j < _cols; j++) {
        result.data!.values[j * _rows + i] = at(i, j);
      }
    }
    return result;
  }

  // Convert the 1D index back to a (from, to) pair
  (int row, int col) fromIndex(int index) {
    int row = index ~/ _cols; // Integer division to get the row (from square)
    int col = index % _cols; // Modulo to get the column (to square)
    return (row, col);
  }
}
