// file: transformer_encoder.dart

import 'dart:math' as math;
import '/nn/module.dart';
import 'transformer_encoder_block.dart'; // Import the new encoder block
import 'layer_norm2.dart';
import '/nn/layer.dart'; // Though not directly used for output, useful if you extend it
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A complete Transformer Encoder model.
///
/// This model takes a sequence of token indices and produces contextualized
/// embeddings for each token. It does not have a language model head for
/// next-token prediction, unlike the decoder-only Transformer.
class TransformerEncoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize; // Maximum context length
  final int numLayers;
  final int numHeads;

  // Learnable parameter tables for token and position embeddings
  final List<ValueVector> tokenEmbeddings;
  final List<ValueVector> positionEmbeddings;

  // Stack of Transformer Encoder blocks
  final List<TransformerEncoderBlock> blocks;

  // Optional final layer normalization (often used in BERT-like encoders)
  final LayerNorm finalLayerNorm;

  TransformerEncoder({
    this.vocabSize = 100, // Default vocabulary size
    this.embedSize = 64, // Default embedding dimension
    this.blockSize = 128, // Default maximum sequence length
    this.numLayers = 6, // Default number of encoder layers
    this.numHeads = 8, // Default number of attention heads
  })  : assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        // Initialize token embeddings with small random values
        tokenEmbeddings = List.generate(
            vocabSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // Initialize position embeddings (also learned)
        positionEmbeddings = List.generate(
            blockSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // Create a stack of encoder blocks
        blocks = List.generate(
            numLayers, (i) => TransformerEncoderBlock(embedSize, numHeads)),
        // Final layer normalization applied after all blocks
        finalLayerNorm = LayerNorm(embedSize);

  /// The forward pass for the Transformer Encoder model.
  ///
  /// Takes a list of integer token indices `idx` and returns a list of
  /// contextualized `ValueVector` embeddings.
  List<ValueVector> forward(List<int> idx) {
    final T = idx.length; // Current sequence length

    if (T > blockSize) {
      throw ArgumentError(
          "Input sequence length ($T) exceeds model's block size ($blockSize). "
          "Consider truncating or padding the input.");
    }

    // 1. Get token and position embeddings and sum them
    // Each input token is converted into an embedding, and positional information is added.
    var x = List.generate(T, (t) {
      final tok_emb = tokenEmbeddings[idx[t]];
      final pos_emb = positionEmbeddings[t];
      return tok_emb + pos_emb;
    });

    // 2. Pass the sequence through all Transformer Encoder blocks
    for (final block in blocks) {
      x = block.forward(x);
    }

    // 3. Apply final layer normalization to the output of the last block
    x = List.generate(T, (t) => finalLayerNorm.forward(x[t]));

    // The output `x` now contains the contextualized embeddings for each input token.
    return x;
  }

  @override
  List<Value> parameters() {
    return [
      ...tokenEmbeddings.expand((vec) => vec.values),
      ...positionEmbeddings.expand((vec) => vec.values),
      ...blocks.expand((block) => block.parameters()),
      ...finalLayerNorm.parameters(),
    ];
  }
}
