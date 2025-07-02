// file: encoder_decoder_transformer.dart

import '/nn/module.dart';
import 'transformer_encoder.dart';
import 'transformer_decoder.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';

/// A full Encoder-Decoder Transformer model for sequence-to-sequence tasks.
///
/// This model uses a TransformerEncoder to process the source sequence
/// and a TransformerDecoder to generate the target sequence, leveraging
/// cross-attention over the encoder's output.
class EncoderDecoderTransformer extends Module {
  final TransformerEncoder encoder;
  final TransformerDecoder decoder;

  EncoderDecoderTransformer({
    required int sourceVocabSize,
    required int targetVocabSize,
    required int embedSize, // Shared embedding size between encoder/decoder
    required int sourceBlockSize,
    required int targetBlockSize,
    required int numLayers,
    required int numHeads,
  })  : encoder = TransformerEncoder(
            vocabSize: sourceVocabSize,
            embedSize: embedSize,
            blockSize: sourceBlockSize,
            numLayers: numLayers,
            numHeads: numHeads),
        // The decoder's encoderEmbedSize must match the encoder's embedSize
        decoder = TransformerDecoder(
            vocabSize: targetVocabSize,
            embedSize: embedSize,
            blockSize: targetBlockSize,
            numLayers: numLayers,
            numHeads: numHeads,
            encoderEmbedSize: embedSize);

  /// Forward pass for the Encoder-Decoder Transformer.
  ///
  /// Takes a `sourceIdx` (input sequence token IDs) and `targetIdx` (output sequence token IDs,
  /// typically shifted for teacher forcing during training).
  /// Returns logits for the next token in the target sequence.
  List<ValueVector> forward(List<int> sourceIdx, List<int> targetIdx) {
    // 1. Encode the source sequence
    final encoderOutput = encoder.forward(sourceIdx); // (source_len, embedSize)

    // 2. Decode the target sequence using the encoder's output
    final decoderLogits = decoder.forward(
        targetIdx, encoderOutput); // (target_len, target_vocab_size)

    return decoderLogits;
  }

  @override
  List<Value> parameters() {
    return [...encoder.parameters(), ...decoder.parameters()];
  }
}
