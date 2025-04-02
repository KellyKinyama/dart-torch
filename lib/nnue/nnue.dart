import 'dart:typed_data';
import 'dart:math';

class NNUE {
  static const int inputSize = 768;
  static const int hiddenSize = 256;
  static const int outputSize = 1;

  // Weight matrices stored as integer arrays
  final Int16List W1 = Int16List(hiddenSize * inputSize);
  final Int16List W2 = Int16List(hiddenSize * hiddenSize);
  final Int16List W3 = Int16List(hiddenSize * hiddenSize);
  final Int16List W4 = Int16List(outputSize * hiddenSize);

  final Int16List hiddenLayer = Int16List(hiddenSize);

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

  // Efficient forward pass
  int evaluate(List<int> pieceSquareInput) {
    // Ensure indices are within valid range
    pieceSquareInput =
        pieceSquareInput.map((x) => x.clamp(0, inputSize - 1)).toList();

    // Compute first hidden layer (W1 * input)
    for (int i = 0; i < hiddenSize; i++) {
      hiddenLayer[i] = 0;
      for (int j = 0; j < pieceSquareInput.length; j++) {
        hiddenLayer[i] += W1[i * inputSize + pieceSquareInput[j]];
      }
      hiddenLayer[i] = max(0, hiddenLayer[i]); // ReLU activation
    }

    // Hidden layers (ReLU(W4 * ReLU(W3 * ReLU(W2 * HiddenLayer))))
    Int16List h1 = _reluMultiply(W2, hiddenLayer);
    Int16List h2 = _reluMultiply(W3, h1);
    Int16List h3 = _reluMultiply(W4, h2, outputSize);

    return h3[0]; // Single output score
  }

  // Efficient matrix multiplication with ReLU
  Int16List _reluMultiply(Int16List weights, Int16List input,
      [int size = hiddenSize]) {
    Int16List output = Int16List(size);
    for (int i = 0; i < size; i++) {
      int sum = 0;
      for (int j = 0; j < hiddenSize; j++) {
        sum += weights[i * hiddenSize + j] * input[j];
      }
      output[i] = max(0, sum); // ReLU
    }
    return output;
  }

  // Incremental update when a move is made
  void update(int removedPiece, int addedPiece) {
    removedPiece = removedPiece.clamp(0, inputSize - 1);
    addedPiece = addedPiece.clamp(0, inputSize - 1);

    for (int i = 0; i < hiddenSize; i++) {
      hiddenLayer[i] -= W1[i * inputSize + removedPiece];
      hiddenLayer[i] += W1[i * inputSize + addedPiece];
      hiddenLayer[i] = max(0, hiddenLayer[i]); // ReLU
    }
  }
}

void main() {
  NNUE nnue = NNUE();
  List<int> boardInput = List.generate(768, (index) => Random().nextInt(768))
      .map((x) => x.clamp(0, NNUE.inputSize - 1))
      .toList();
  int score = nnue.evaluate(boardInput);
  print('Initial Score: $score');

  // Simulate a move update
  nnue.update(boardInput[0], boardInput[10]);
  score = nnue.evaluate(boardInput);
  print('Updated Score: $score');
}
