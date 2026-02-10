import 'module.dart';
import 'tensor.dart';

class Embedding extends Module {
  final Tensor weight;
  final int vocabSize;
  final int embedSize;

  Embedding(this.vocabSize, this.embedSize)
      : weight = Tensor.random([vocabSize, embedSize]);

  /// Takes a list of token IDs and returns a [T, EmbedSize] Tensor
  Tensor forward(List<int> tokens) {
    final int T = tokens.length;
    final out = Tensor([T, embedSize], children: {weight});

    // Forward Pass: Copy rows from the weight matrix
    for (int t = 0; t < T; t++) {
      int id = tokens[t];
      // Safety check for vocab bounds
      if (id < 0 || id >= vocabSize) id = 0;

      for (int i = 0; i < embedSize; i++) {
        out.data[t * embedSize + i] = weight.data[id * embedSize + i];
      }
    }

    // Use the public setter 'onBackward' to assign the gradient logic
    out.onBackward = () {
      for (int t = 0; t < T; t++) {
        int id = tokens[t];
        for (int i = 0; i < embedSize; i++) {
          // Accumulate gradients back into the specific word's row in the weight matrix
          weight.grad[id * embedSize + i] += out.grad[t * embedSize + i];
        }
      }
    };

    return out;
  }

  @override
  List<Tensor> parameters() => [weight];
}
