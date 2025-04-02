import 'dart:typed_data';
import 'dart:math';

class NNUE {
  static const int inputSize = 41024; // HalfKP structure
  static const int hiddenSize = 256;
  static const int interSize = 32;
  static const int outputSize = 1;

  // Weight matrices stored as integer arrays
  final Int16List W1 = Int16List(hiddenSize * inputSize);
  final Int16List W2 = Int16List(hiddenSize * interSize);
  final Int16List W3 = Int16List(interSize * interSize);
  final Int16List W4 = Int16List(outputSize * interSize);

  final Int16List hiddenLayer = Int16List(hiddenSize);
  final Int16List interLayer = Int16List(interSize);

  NNUE() {
    _initializeWeights();
  }

  void _initializeWeights() {
    final random = Random();
    for (int i = 0; i < W1.length; i++) {
      W1[i] = (random.nextDouble() * 100).toInt() - 50;
    }
    for (int i = 0; i < W2.length; i++) {
      W2[i] = (random.nextDouble() * 100).toInt() - 50;
    }
    for (int i = 0; i < W3.length; i++) {
      W3[i] = (random.nextDouble() * 100).toInt() - 50;
    }
    for (int i = 0; i < W4.length; i++) {
      W4[i] = (random.nextDouble() * 100).toInt() - 50;
    }
  }

  int evaluate(List<int> pieceSquareInput) {
    pieceSquareInput =
        pieceSquareInput.map((x) => x.clamp(0, inputSize - 1)).toList();

    for (int i = 0; i < hiddenSize; i++) {
      hiddenLayer[i] = 0;
      for (int j = 0; j < pieceSquareInput.length; j++) {
        hiddenLayer[i] += W1[i * inputSize + pieceSquareInput[j]];
      }
      hiddenLayer[i] = max(0, hiddenLayer[i]); // ReLU activation
    }

    Int16List h1 = _reluMultiply(W2, hiddenLayer, interSize, hiddenSize);
    Int16List h2 = _reluMultiply(W3, h1, interSize, interSize);
    Int16List h3 = _reluMultiply(W4, h2, outputSize, interSize);

    return h3[0];
  }

  Int16List _reluMultiply(
      Int16List weights, Int16List input, int size, int inputSize) {
    Int16List output = Int16List(size);
    for (int i = 0; i < size; i++) {
      int sum = 0;
      for (int j = 0; j < inputSize; j++) {
        sum += weights[i * inputSize + j] * input[j];
      }
      output[i] = max(0, sum & 0xFFFF); // ReLU and clamp to 16-bit
    }
    return output;
  }

  void update(int removedPiece, int addedPiece) {
    removedPiece = removedPiece.clamp(0, inputSize - 1);
    addedPiece = addedPiece.clamp(0, inputSize - 1);

    for (int i = 0; i < hiddenSize; i++) {
      hiddenLayer[i] -= W1[i * inputSize + removedPiece];
      hiddenLayer[i] += W1[i * inputSize + addedPiece];
      hiddenLayer[i] = max(0, hiddenLayer[i]);
    }
  }
}

void main() {
  NNUE nnue = NNUE();
  List<int> boardInput =
      List.generate(41024, (index) => Random().nextInt(41024))
          .map((x) => x.clamp(0, NNUE.inputSize - 1))
          .toList();
  int score = nnue.evaluate(boardInput);
  print('Initial Score: $score');

  nnue.update(boardInput[0], boardInput[10]);
  score = nnue.evaluate(boardInput);
  print('Updated Score: $score');
}
