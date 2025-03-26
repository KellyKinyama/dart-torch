// import 'encoder.dart';
import 'linear_layer.dart';
import 'layer_norm.dart';
import '../nn/value_vector.dart';
import 'multi_head_attention.dart';

class TransformerBlock {
  final MultiHeadAttention attention;
  final LinearLayer ff1;
  final LinearLayer ff2;
  final LayerNorm ln1;
  final LayerNorm ln2;

  TransformerBlock(int embedDim, int numHeads, int ffDim)
      : attention = MultiHeadAttention(embedDim, numHeads),
        ff1 = LinearLayer(embedDim, ffDim),
        ff2 = LinearLayer(ffDim, embedDim),
        ln1 = LayerNorm(embedDim),
        ln2 = LayerNorm(embedDim);

  List<ValueVector> forward(List<ValueVector> x) {
    // Normalize input before attention
    final normX = x.map((v) => ln1.forward(v)).toList();

    // Self-attention
    final attnOut = attention.forward(normX);

    // Residual connection + normalization
    final x1 = List.generate(x.length, (i) => x[i] + attnOut[i]);
    final normX1 = x1.map((v) => ln2.forward(v)).toList();

    // Feed-forward network
    final List<ValueVector> ffOut = [];
    for (final vec in normX1) {
      final ff = ff2.forward(ff1.forward(vec).reLU());
      ffOut.add(ff);
    }

    // Second residual connection
    return List.generate(x.length, (i) => x1[i] + ffOut[i]);
  }

  @override
  String toString() => 'TransformerBlock()';
}
