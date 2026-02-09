// file: example_aft_generation.dart

import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart'; // Ensure this uses your AFT-based blocks

void main() {
  print("--- Attention Free Transformer (AFT) Generation Example ---");

  // 1. Define AFT Model Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 32;
  const int blockSize = 10; // Fixed size for AFT position bias matrix
  const int numLayers = 3;
  const int numHeads = 4;

  // 2. Simple Vocabulary (unchanged)
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
  final int startTokenId = stoi["<start>"]!;
  final int endTokenId = stoi["."]!;

  // 3. Instantiate the AFT model
  print("\nInitializing AFT-based TransformerDecoder...");
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize:
        blockSize, // CRITICAL: This defines the T dimension of AFT biases
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );

  print("AFT Model initialized. Parameters: ${gptModel.parameters().length}");

  // 4. Text Generation Loop (Greedy Sampling)
  print("\n--- Starting Text Generation ---");
  List<int> generatedSequence = [startTokenId];
  final int maxGenerationLength = 15;

  // Dummy context for cross-attention compatibility
  final List<ValueVector> simpleDummyEncoderOutput = [
    ValueVector(List.filled(embedSize, Value(0.0)))
  ];

  for (int i = 0; i < maxGenerationLength; i++) {
    // Truncate to blockSize. AFT cannot look back further than its w matrix size.
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    // Forward pass: AFT uses element-wise weighting instead of softmax(QK^T)
    final List<ValueVector> logits =
        gptModel.forward(currentInput, simpleDummyEncoderOutput);

    // Get logits for the last token
    final ValueVector lastTokenLogits = logits.last;

    // Greedy sampling
    double maxProb = -1.0;
    int predictedNextToken = -1;
    final ValueVector probabilities = lastTokenLogits.softmax();

    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    generatedSequence.add(predictedNextToken);
    print("Generated: ${generatedSequence.map((id) => itos[id]).join(' ')}");

    if (predictedNextToken == endTokenId) break;
  }

  print("\n--- Final Generated Sequence ---");
  print(generatedSequence.map((id) => itos[id]).join(' '));
}
