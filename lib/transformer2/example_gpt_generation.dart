// file: example_gpt_generation.dart

import 'dart:math';

// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/module.dart';
import '/nn/layer.dart'; // Assuming your Layer is in nn/layer.dart

// Import your TransformerDecoder
import 'transformer_decoder.dart'; // Your existing TransformerDecoder
// Also ensure transformer_decoder_block.dart, multi_head_attention.dart,
// feed_forward.dart, layer_norm2.dart, self_attention.dart are accessible.
// NOTE: Your TransformerDecoderBlock currently has a MultiHeadCrossAttention.
// For a pure GPT, this cross-attention layer would typically be removed
// as there's no encoder output to attend to.
// For this example, we will pass a dummy encoderOutput to satisfy the current
// TransformerDecoderBlock's forward method.
// A more accurate GPT would involve modifying transformer_decoder_block.dart
// to either remove MultiHeadCrossAttention or make it conditional.
import 'multi_head_cross_attention.dart'; // Needed due to current TransformerDecoderBlock
import 'cross_attention.dart'; // Needed due to current TransformerDecoderBlock

void main() {
  print("--- Generative Pretrained Transformer (GPT) Example ---");

  // 1. Define GPT Model Hyperparameters
  const int vocabSize =
      20; // Example vocabulary size (e.g., a few common words)
  const int embedSize = 32;
  const int blockSize = 10; // Maximum sequence length the GPT can process
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  // 2. Simple Vocabulary for demonstration
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
    ".": 16, // End of sentence token
    "<start>": 17, // Start of sequence token
    "<pad>": 18, // Padding token
    // You might also have an <unk> token for unknown words
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // Get the ID for the start token
  final int startTokenId = stoi["<start>"]!;
  final int endTokenId = stoi["."]!; // Using '.' as an example end token

  print("\nExample Vocabulary:");
  print(itos);

  // 3. Instantiate the GPT model (your TransformerDecoder)
  print("\nInitializing GPT (TransformerDecoder)...");
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    // For a GPT, the cross-attention part of TransformerDecoderBlock is not used.
    // We pass embedSize here just to satisfy the constructor.
    // In a pure GPT, you'd likely have a separate TransformerDecoder class
    // that doesn't include cross-attention at all.
    encoderEmbedSize: embedSize,
  );
  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 4. Text Generation Loop (Greedy Sampling)
  print("\n--- Starting Text Generation ---");
  List<int> generatedSequence = [startTokenId]; // Start with the <start> token
  final int maxGenerationLength = 15; // Max tokens to generate

  // Create a dummy encoder output for the cross-attention layer in TransformerDecoderBlock.
  // In a true GPT, the cross-attention layer would not exist, or its input would be ignored.
  // Here, we provide an empty list or a list of zeros to prevent errors,
  // knowing that the masked self-attention is what's truly driving generation.
  final List<ValueVector> simpleDummyEncoderOutput = [
    ValueVector(List.filled(embedSize, Value(0.0)))
  ]; // (1, embedSize)

  for (int i = 0; i < maxGenerationLength; i++) {
    // If the sequence exceeds blockSize, truncate it (common for long contexts)
    // Or, for generation, keep expanding and handle attention efficiently.
    // For simplicity, we'll keep the whole generated sequence for now if within blockSize.
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    // Forward pass through the GPT (TransformerDecoder)
    // Pass the dummy encoder output to satisfy the method signature.
    final List<ValueVector> logits =
        gptModel.forward(currentInput, simpleDummyEncoderOutput);

    // Get the logits for the *last* token in the sequence (the prediction for the next token)
    final ValueVector lastTokenLogits = logits.last;

    // Apply softmax to get probabilities
    final ValueVector probabilities = lastTokenLogits.softmax();

    // Greedy sampling: pick the token with the highest probability
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    // Add the predicted token to the generated sequence
    generatedSequence.add(predictedNextToken);

    // Print current generation progress (convert IDs back to words)
    print("Generated: ${generatedSequence.map((id) => itos[id]).join(' ')}");

    // Stop if an end-of-sequence token is predicted
    if (predictedNextToken == endTokenId) {
      print("End of sequence token detected.");
      break;
    }
    if (generatedSequence.length >= maxGenerationLength + 1) {
      // +1 because we start with <start>
      print("Maximum generation length reached.");
      break;
    }
  }

  print("\n--- Final Generated Sequence ---");
  print(generatedSequence.map((id) => itos[id]).join(' '));
  print("--------------------------------");
}
