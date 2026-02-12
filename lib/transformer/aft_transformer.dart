// file: transformer.dart

import 'dart:math' as math;
import '/nn/module.dart';
import 'aft_transformer_block.dart';
import 'layer_norm2.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A complete decoder-only Transformer model (GPT-style).
class Transformer extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize; // Maximum context length
  final int numLayers;
  final int numHeads;

  // Learnable parameter tables
  final List<ValueVector> tokenEmbeddings;
  final List<ValueVector> positionEmbeddings;

  // Network layers
  final List<TransformerBlock> blocks;
  final LayerNorm finalLayerNorm;
  final Layer lmHead; // Language Model Head projects to vocab size

  Transformer({
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.numLayers = 4,
    this.numHeads = 4,
  })  :
        // Learnable embedding tables
        tokenEmbeddings = List.generate(
            vocabSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        positionEmbeddings = List.generate(
            blockSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),

        // Stack of transformer blocks
        blocks = List.generate(
            numLayers,
            (i) =>
                TransformerBlock(embedSize, numHeads, blockSize, masked: true)),

        // Final layers for output
        finalLayerNorm = LayerNorm(embedSize),
        lmHead = Layer.fromNeurons(embedSize, vocabSize);

  /// The forward pass for the entire Transformer model.
  ///
  /// Takes a list of integer token indices `idx` and returns logits for the next token.
  List<ValueVector> forward(List<int> idx) {
    final T = idx.length;
    if (T > blockSize) {
      throw ArgumentError(
          "Input sequence length ($T) exceeds model's block size ($blockSize)");
    }

    // 1. Get token and position embeddings and sum them
    var x = List.generate(T, (t) {
      final tok_emb = tokenEmbeddings[idx[t]];
      final pos_emb = positionEmbeddings[t];
      return tok_emb + pos_emb;
    });

    // 2. Pass sequence through all transformer blocks
    for (final block in blocks) {
      x = block.forward(x);
    }

    // 3. Apply final layer norm
    x = List.generate(T, (t) => finalLayerNorm.forward(x[t]));

    // 4. Language model head to get final logits
    final logits = List.generate(T, (t) => lmHead.forward(x[t]));

    return logits;
  }

  @override
  List<Value> parameters() {
    return [
      ...tokenEmbeddings.expand((v) => v.values),
      ...positionEmbeddings.expand((v) => v.values),
      ...blocks.expand((b) => b.parameters()),
      ...finalLayerNorm.parameters(),
      ...lmHead.parameters(),
    ];
  }
}
