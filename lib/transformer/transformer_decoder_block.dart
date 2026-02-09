// // file: transformer_decoder_block.dart (RENAMED from transformer_block.dart)

// import '/nn/module.dart';
// import 'multi_head_attention.dart'; // For self-attention
// import 'multi_head_cross_attention.dart'; // NEW: For cross-attention
// import 'feed_forward.dart';
// import 'layer_norm2.dart';
// import '/nn/value.dart';
// import '/nn/value_vector.dart';

// /// A single Transformer Decoder block.
// ///
// /// This block combines masked Multi-Head Self-Attention, Multi-Head Cross-Attention,
// /// and a Feed-Forward Network, each with residual connections and layer normalization.
// class TransformerDecoderBlock extends Module {
//   final MultiHeadAttention selfAttention;
//   final MultiHeadCrossAttention crossAttention; // NEW
//   final FeedForward ffn;
//   final LayerNorm ln1; // LayerNorm before self-attention
//   final LayerNorm ln2; // LayerNorm before cross-attention (NEW)
//   final LayerNorm ln3; // LayerNorm before FFN (formerly ln2)
//   final int embedSize;

//   TransformerDecoderBlock(this.embedSize, int numHeads,
//       int encoderEmbedSize) // NEW: encoderEmbedSize for cross-attention
//       : selfAttention = MultiHeadAttention(numHeads, embedSize,
//             masked: true), // Masked self-attention
//         crossAttention = MultiHeadCrossAttention(
//             numHeads, embedSize, encoderEmbedSize), // NEW
//         ffn = FeedForward(embedSize),
//         ln1 = LayerNorm(embedSize),
//         ln2 = LayerNorm(embedSize), // NEW
//         ln3 = LayerNorm(embedSize); // Renamed

//   /// The forward pass through a single Transformer Decoder block.
//   ///
//   /// Takes decoder input `x_decoder` and encoder output `x_encoder`.
//   List<ValueVector> forward(
//       List<ValueVector> x_decoder, List<ValueVector> x_encoder) {
//     final T_dec = x_decoder.length; // Decoder sequence length

//     // 1. First sub-layer: Masked Multi-Head Self-Attention
//     final x_norm1 = List.generate(T_dec, (i) => ln1.forward(x_decoder[i]));
//     final self_attn_out = selfAttention.forward(x_norm1);
//     final x_res1 = List.generate(T_dec, (i) => x_decoder[i] + self_attn_out[i]);

//     // 2. Second sub-layer: Multi-Head Cross-Attention (NEW)
//     final x_norm2 = List.generate(T_dec, (i) => ln2.forward(x_res1[i]));
//     final cross_attn_out = crossAttention.forward(
//         x_norm2, x_encoder); // Pass both decoder input and encoder output
//     final x_res2 = List.generate(T_dec, (i) => x_res1[i] + cross_attn_out[i]);

//     // 3. Third sub-layer: Feed-Forward Network
//     final x_norm3 =
//         List.generate(T_dec, (i) => ln3.forward(x_res2[i])); // Renamed from ln2
//     final ffn_out = List.generate(T_dec, (i) => ffn.forward(x_norm3[i]));
//     final x_res3 = List.generate(T_dec, (i) => x_res2[i] + ffn_out[i]);

//     return x_res3;
//   }

//   @override
//   List<Value> parameters() {
//     return [
//       ...selfAttention.parameters(),
//       ...crossAttention.parameters(), // NEW
//       ...ffn.parameters(),
//       ...ln1.parameters(),
//       ...ln2.parameters(), // NEW
//       ...ln3.parameters(), // Renamed
//     ];
//   }
// }
