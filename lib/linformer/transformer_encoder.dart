// file: transformer_encoder.dart (MODIFIED)

import 'dart:math' as math;
import '/nn/module.dart';
import 'transformer_encoder_block.dart';
import '../transformer/layer_norm2.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A complete Transformer Encoder model.
class TransformerEncoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize;
  final int numLayers;
  final int numHeads;
  final int? projK; // NEW: Add projK field

  final List<ValueVector> tokenEmbeddings;
  final List<ValueVector> positionEmbeddings;
  final List<TransformerEncoderBlock> blocks;
  final LayerNorm finalLayerNorm;

  TransformerEncoder({
    this.vocabSize = 100,
    this.embedSize = 64,
    this.blockSize = 128,
    this.numLayers = 6,
    this.numHeads = 8,
    this.projK = 128, // NEW: Add projK to the constructor with a default value
  })  : assert(embedSize % numHeads == 0,
            "embedSize must be divisible by numHeads"),
        tokenEmbeddings = List.generate(
            vocabSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        positionEmbeddings = List.generate(
            blockSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // CORRECTED: Pass the projK parameter to the TransformerEncoderBlock constructor
        blocks = List.generate(numLayers,
            (i) => TransformerEncoderBlock(embedSize, numHeads, projK: projK)),
        finalLayerNorm = LayerNorm(embedSize);

  /// Standard forward pass for token IDs (used in NLP tasks).
  List<ValueVector> forward(List<int> idx) {
    final T = idx.length;

    if (T > blockSize) {
      throw ArgumentError(
          "Input sequence length ($T) exceeds model's block size ($blockSize). "
          "Consider truncating or padding the input.");
    }

    var x = List.generate(T, (t) {
      final tok_emb = tokenEmbeddings[idx[t]];
      final pos_emb = positionEmbeddings[t];
      return tok_emb + pos_emb;
    });

    return _forwardThroughBlocks(x);
  }

  /// NEW: Forward pass for already pre-computed embeddings (used in ViT).
  List<ValueVector> forwardEmbeddings(List<ValueVector> embeddedInputs) {
    if (embeddedInputs.length > blockSize) {
      throw ArgumentError(
          "Input sequence length (${embeddedInputs.length}) exceeds model's block size ($blockSize). "
          "Consider truncating or padding the input.");
    }
    return _forwardThroughBlocks(embeddedInputs);
  }

  List<ValueVector> _forwardThroughBlocks(List<ValueVector> x) {
    for (final block in blocks) {
      x = block.forward(x);
    }
    x = List.generate(x.length, (t) => finalLayerNorm.forward(x[t]));
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
