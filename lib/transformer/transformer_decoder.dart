// file: transformer_decoder.dart (RENAMED from transformer.dart)

import 'dart:math' as math;
import '/nn/module.dart';
import 'aft_transformer_decoder_block.dart';
// import 'transformer_decoder_block.dart'; // Import the new decoder block
import 'layer_norm2.dart';
import '/nn/layer.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A Transformer Decoder model.
///
/// This model takes a target sequence and encoder output, and produces
/// logits for the next token in the target sequence.
class TransformerDecoder extends Module {
  final int vocabSize;
  final int embedSize;
  final int blockSize; // Maximum context length for the decoder
  final int numLayers;
  final int numHeads;
  final int encoderEmbedSize; // NEW: The embedding size of the encoder's output

  // Learnable parameter tables
  final List<ValueVector> tokenEmbeddings;
  final List<ValueVector> positionEmbeddings;

  // Network layers
  final List<TransformerDecoderBlock> blocks; // Uses the new decoder block
  final LayerNorm finalLayerNorm;
  final Layer lmHead; // Language Model Head projects to vocab size

  TransformerDecoder({
    this.vocabSize = 50,
    this.embedSize = 32,
    this.blockSize = 8,
    this.numLayers = 4,
    this.numHeads = 4,
    this.encoderEmbedSize = 64, // NEW: Must match the encoder's embedSize
  })  : assert(embedSize % numHeads == 0),
        tokenEmbeddings = List.generate(
            vocabSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        positionEmbeddings = List.generate(
            blockSize,
            (i) => ValueVector.fromDoubleList(List.generate(
                embedSize, (j) => math.Random().nextDouble() * 0.02 - 0.01))),
        // Each decoder block needs the encoder's embed size for cross-attention
        blocks = List.generate(
            numLayers,
            (i) => TransformerDecoderBlock(
                  embedSize,
                  numHeads,
                  encoderEmbedSize,
                  blockSize,
                )),
        finalLayerNorm = LayerNorm(embedSize),
        lmHead = Layer.fromNeurons(embedSize, vocabSize);

  /// The forward pass for the Transformer Decoder model.
  ///
  /// Takes a list of integer target token indices `idx` (shifted right for training)
  /// and the `encoderOutput` from the Transformer Encoder.
  /// Returns logits for the next token prediction.
  List<ValueVector> forward(List<int> idx, List<ValueVector> encoderOutput) {
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

    // 2. Pass sequence through all transformer decoder blocks
    // Each block now also receives the encoder's output
    for (final block in blocks) {
      x = block.forward(x, encoderOutput); // Pass encoderOutput
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
      ...tokenEmbeddings.expand((vec) => vec.values),
      ...positionEmbeddings.expand((vec) => vec.values),
      ...blocks.expand((block) => block.parameters()),
      ...finalLayerNorm.parameters(),
      ...lmHead.parameters(),
    ];
  }
}
