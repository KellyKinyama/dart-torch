import 'dart:convert';
import 'dart:io';
import 'dart:math';
import '../nn/value.dart';
import '../nn/value_vector.dart';
import 'linear_layer.dart';
import 'layer_norm.dart';
import 'multi_head_attention.dart';
import 'transformer_block.dart';
import 'encoder.dart';
import 'decoder.dart';

class Transformer {
  final TransformerEncoder encoder;
  final TransformerDecoder decoder;

  Transformer(int numLayers, int embedDim, int numHeads, int ffDim)
      : encoder = TransformerEncoder(numLayers, embedDim, numHeads),
        decoder = TransformerDecoder(numLayers, embedDim, numHeads, ffDim);

  List<ValueVector> forward(List<ValueVector> input, List<ValueVector> target) {
    // Encoder forward pass
    final encoderOutput = List.generate(
        input.length, (int index) => encoder.forward(input[index], input));

    // Decoder forward pass
    return decoder.call(target, encoderOutput);
  }
}

void main() async {
  // Define hyperparameters for the Transformer model
  final numLayers = 6; // Number of layers in encoder/decoder
  final embedDim = 512; // Dimensionality of embeddings
  final numHeads = 8; // Number of attention heads
  final ffDim = 2048; // Dimensionality of feed-forward network

  // Create a sample input sequence and target sequence
  final input = [
    ValueVector([Value(1.0), Value(2.0), Value(3.0), Value(4.0)]),
    ValueVector([Value(2.0), Value(3.0), Value(4.0), Value(5.0)]),
    ValueVector([Value(3.0), Value(4.0), Value(5.0), Value(6.0)]),
  ];
  final target = [
    ValueVector([Value(0.5), Value(1.0), Value(1.5), Value(2.0)]),
    ValueVector([Value(1.0), Value(1.5), Value(2.0), Value(2.5)]),
    ValueVector([Value(1.5), Value(2.0), Value(2.5), Value(3.0)]),
  ];

  // Initialize the Transformer model
  final transformer = Transformer(numLayers, embedDim, numHeads, ffDim);

  // Forward pass through the Transformer
  final output = transformer.forward(input, target);

  // Print the output from the Transformer model
  print("Transformer Output:");
  for (var vec in output) {
    print(vec.values);
  }
}

// class TransformerEncoder {
//   final List<TransformerBlock> layers;
//   final LayerNorm finalNorm;

//   TransformerEncoder(int numLayers, int embedDim, int numHeads, int ffDim)
//       : layers = List.generate(
//             numLayers,
//             (_) => TransformerBlock(embedDim, numHeads, ffDim)),
//         finalNorm = LayerNorm(embedDim);

//   List<ValueVector> forward(List<ValueVector> x) {
//     for (var layer in layers) {
//       x = layer.forward(x);
//     }
//     return x.map((v) => finalNorm.forward(v)).toList();
//   }
// }

// class TransformerDecoder {
//   final List<TransformerBlock> layers;
//   final LayerNorm finalNorm;

//   TransformerDecoder(int numLayers, int embedDim, int numHeads, int ffDim)
//       : layers = List.generate(
//             numLayers,
//             (_) => TransformerBlock(embedDim, numHeads, ffDim)),
//         finalNorm = LayerNorm(embedDim);

//   List<ValueVector> call(List<ValueVector> x, List<ValueVector> encoderOutput) {
//     for (var layer in layers) {
//       x = layer.forward(x);
//     }
//     return x.map((v) => finalNorm.forward(v)).toList();
//   }
// }

// class TransformerBlock {
//   final MultiHeadAttention attention;
//   final LinearLayer ff1;
//   final LinearLayer ff2;
//   final LayerNorm ln1;
//   final LayerNorm ln2;

//   TransformerBlock(int embedDim, int numHeads, int ffDim)
//       : attention = MultiHeadAttention(embedDim, numHeads),
//         ff1 = LinearLayer(embedDim, ffDim),
//         ff2 = LinearLayer(ffDim, embedDim),
//         ln1 = LayerNorm(embedDim),
//         ln2 = LayerNorm(embedDim);

//   List<ValueVector> forward(List<ValueVector> x) {
//     // Normalize input before attention
//     final normX = x.map((v) => ln1.forward(v)).toList();

//     // Self-attention
//     final attnOut = attention.forward(normX);

//     // Residual connection + normalization
//     final x1 = List.generate(x.length, (i) => x[i] + attnOut[i]);
//     final normX1 = x1.map((v) => ln2.forward(v)).toList();

//     // Feed-forward network
//     final List<ValueVector> ffOut = [];
//     for (final vec in normX1) {
//       final ff = ff2.forward(ff1.forward(vec).reLU());
//       ffOut.add(ff);
//     }

//     // Second residual connection
//     return List.generate(x.length, (i) => x1[i] + ffOut[i]);
//   }

//   @override
//   String toString() => 'TransformerBlock()';
// }
