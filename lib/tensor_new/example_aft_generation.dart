import '../tensor/tensor.dart';
import 'aft_transformer_decoder.dart';

void main() {
  print("--- Tensor-Based Attention Free Transformer (AFT) Generation ---");

  // 1. Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 32;
  const int blockSize = 10;
  const int numLayers = 3;
  const int numHeads = 4;

  // 2. Vocabulary
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "generation": 6,
    "model": 7,
    "the": 8,
    "quick": 9,
    "brown": 10,
    "fox": 11,
    "jumps": 12,
    "over": 13,
    "lazy": 14,
    "dog": 15,
    ".": 16,
    "<start>": 17,
    "<pad>": 18,
  };
  final itos = stoi.map((k, v) => MapEntry(v, k));
  final int startTokenId = stoi["<start>"]!;
  final int endTokenId = stoi["."]!;

  // 3. Initialize Model
  final model = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );

  // Dummy Encoder Output [1, embedSize] for cross-attention
  final encoderOutput = Tensor.zeros([1, embedSize]);

  // 4. Generation Loop
  List<int> generatedIndices = [startTokenId];
  const int maxLen = 12;

  print("\nGenerating...");
  for (int i = 0; i < maxLen; i++) {
    // AFT Context window constraint
    List<int> inputIndices = generatedIndices.length > blockSize
        ? generatedIndices.sublist(generatedIndices.length - blockSize)
        : generatedIndices;

    // Forward pass: returns [T, vocabSize]
    Tensor logits = model.forward(inputIndices, encoderOutput);

    // Get the last row of logits (prediction for the next token)
    // We use getRow(T-1) to isolate the most recent time-step
    Tensor lastTokenLogits = logits.getRow(inputIndices.length - 1);

    // Get the index of the maximum value (Greedy Search)
    int nextToken = _argmax(lastTokenLogits);

    generatedIndices.add(nextToken);

    // Live update
    print("Step ${i + 1}: ${generatedIndices.map((id) => itos[id]).join(' ')}");

    if (nextToken == endTokenId) break;
  }

  print("\n--- Final Output ---");
  print(generatedIndices.map((id) => itos[id]).join(' '));
}

/// Helper function to find the index of the highest value in a Tensor
int _argmax(Tensor t) {
  double maxVal = double.negativeInfinity;
  int maxIdx = 0;
  for (int i = 0; i < t.data.length; i++) {
    if (t.data[i] > maxVal) {
      maxVal = t.data[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}
