// file: multi_head_cross_attention.dart

// import '../transformer/cross_attention.dart';
import '/nn/module.dart';
import 'cross_attention.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import '/nn/layer.dart';



/// Combines multiple cross-attention heads to run in parallel.
class MultiHeadCrossAttention extends Module {
  // The list holds the specific type 'CrossAttention' which can also contain its subtype 'LinformerCrossAttention'.
  final List<CrossAttention> heads;
  final Layer proj; // Final linear projection layer
  final int numHeads;
  final int headSize;
  final int? projK;

  MultiHeadCrossAttention(
      this.numHeads, int decoderEmbedSize, int encoderEmbedSize, {this.projK})
      : assert(decoderEmbedSize % numHeads == 0),
        headSize = decoderEmbedSize ~/ numHeads,
        heads = projK != null
            ? List.generate(
                numHeads,
                (i) => LinformerCrossAttention(
                    decoderEmbedSize, encoderEmbedSize, decoderEmbedSize ~/ numHeads,
                    projK: projK!))
            : List.generate(
                numHeads,
                (i) => CrossAttention(
                    decoderEmbedSize, encoderEmbedSize, decoderEmbedSize ~/ numHeads)),
        proj = Layer.fromNeurons(decoderEmbedSize, decoderEmbedSize);

  /// Forward pass for Multi-Head Cross Attention.
  List<ValueVector> forward(
      List<ValueVector> x_decoder, List<ValueVector> x_encoder) {
    // 1. Compute all head outputs in parallel
    final headOutputs = heads.map((h) => h.forward(x_decoder, x_encoder)).toList();

    // 2. Concatenate head outputs for each token position in the decoder sequence
    final T_dec = x_decoder.length;
    final concatenated = List.generate(T_dec, (i) {
      final values =
          headOutputs.expand((head_out) => head_out[i].values).toList();
      return ValueVector(values);
    });

    // 3. Apply final projection layer
    final out = concatenated.map((c) => proj.forward(c)).toList();
    return out;
  }

  @override
  List<Value> parameters() {
    return [...heads.expand((h) => h.parameters()), ...proj.parameters()];
  }
}