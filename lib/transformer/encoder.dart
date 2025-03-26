import 'multi_head_attention.dart';
import 'linear_layer.dart';
import 'layer_norm.dart';
import '../nn/value_vector.dart';

class TransformerEncoder {
  final int dim;
  final int heads;
  final int ffDim;
  final MultiHeadAttention mha;
  final LinearLayer ff1;
  final LinearLayer ff2;
  final LayerNorm norm1;
  final LayerNorm norm2;

  TransformerEncoder(this.dim, this.heads, this.ffDim)
      : mha = MultiHeadAttention(dim, heads),
        ff1 = LinearLayer(dim, ffDim),
        ff2 = LinearLayer(ffDim, dim),
        norm1 = LayerNorm(dim),
        norm2 = LayerNorm(dim);

  ValueVector forward(ValueVector x, List<ValueVector> context) {
    // Pass both input and context through the multi-head attention
    final attn =
        mha.forward([x] + context); // Convert x to a list and append context

    // Residual connection + normalization
    final out1 = norm1.forward(x + attn.first); // Using the first head's output

    // Feed-forward network with ReLU activation
    final ffOut = ff2.forward(ff1.forward(out1).reLU());

    // Second residual connection + normalization
    return norm2.forward(out1 + ffOut);
  }
}

// void main() {
//   // Create a sample input vector (e.g., embedding of a token or sequence)
//   final input = ValueVector([1.0, 2.0, 3.0, 4.0]);

//   // Create context for attention (for simplicity, using the same input for context)
//   final context = [ValueVector([1.0, 2.0, 3.0, 4.0])];

//   // Create the TransformerEncoder
//   final encoder = TransformerEncoder(4, 2, 8);  // dim=4, heads=2, ffDim=8

//   // Forward pass through the encoder
//   final output = encoder.forward(input, context);

//   // Print the output (it will just return the input due to simplified components)
//   print("Encoder Output: ${output.values}");
// }
