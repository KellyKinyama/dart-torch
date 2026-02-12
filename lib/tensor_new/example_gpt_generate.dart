import 'dart:math' as math;
import '../tensor/tensor.dart';
import 'transformer_decoder.dart';

void main() {
  print("--- Tensor-Based GPT Generation Example ---");

  // 1. Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 32;
  const int blockSize = 10;
  const int numLayers = 3;
  const int numHeads = 4;

  // 2. Vocabulary Mapping
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
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // 3. Initialize Model
  // Note: We use the weights learned during training (or random for this example)
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  // 4. Generation Setup
  final int startTokenId = stoi["<start>"]!;
  final int endTokenId = stoi["."]!;
  List<int> generatedSequence = [startTokenId];
  const int maxGenerationLength = 15;

  // Dummy encoder output [1, embedSize] to satisfy cross-attention if present
  final dummyEncoder = Tensor.zeros([1, embedSize]);

  print("\n--- Starting Autoregressive Generation ---");

  for (int i = 0; i < maxGenerationLength; i++) {
    // A. Prepare Context (Truncate to blockSize if sequence is too long)
    List<int> currentContext = generatedSequence;
    if (currentContext.length > blockSize) {
      currentContext =
          currentContext.sublist(currentContext.length - blockSize);
    }

    // B. Forward Pass (No gradients needed for inference)
    // logits shape: [currentContext.length, vocabSize]
    final logits = gptModel.forward(currentContext, dummyEncoder);

    // C. Get Logits for the LAST token in the sequence
    // The last row of our 2D logits tensor contains the next-token prediction
    int lastRowIndex = currentContext.length - 1;
    int offset = lastRowIndex * vocabSize;

    // D. Greedy Sampling (Pick the ID with the highest logit value)
    int predictedNextToken = -1;
    double maxVal = -double.infinity;

    for (int v = 0; v < vocabSize; v++) {
      double val = logits.data[offset + v];
      if (val > maxVal) {
        maxVal = val;
        predictedNextToken = v;
      }
    }

    // E. Update Sequence
    generatedSequence.add(predictedNextToken);

    // Visual Feedback
    String word = itos[predictedNextToken] ?? "???";
    print("Step ${i + 1}: Predicted '$word' (ID: $predictedNextToken)");

    // F. Termination conditions
    if (predictedNextToken == endTokenId) {
      print("\n[End of sequence token reached]");
      break;
    }
  }

  print("\n--- Final Generated Result ---");
  print(generatedSequence.map((id) => itos[id] ?? "???").join(' '));
  print("-------------------------------");
}
