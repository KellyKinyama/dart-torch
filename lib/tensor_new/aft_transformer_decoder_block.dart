import '../tensor/tensor.dart';
import 'adam.dart';
import 'aft_multi_head_attention.dart';
import 'aft_multi_head_cross.dart';
import 'feed_forward.dart';
import 'layer_norm.dart';
import 'module.dart';

class TransformerDecoderBlock extends Module {
  final MultiHeadAFT selfAttention;
  final MultiHeadAFTCross crossAttention;
  final FeedForward ffn;
  final LayerNorm ln1;
  final LayerNorm ln2;
  final LayerNorm ln3;
  final int embedSize;

  TransformerDecoderBlock(
      this.embedSize, int numHeads, int encoderEmbedSize, int maxSeqLen)
      : selfAttention =
            MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: true),
        crossAttention = MultiHeadAFTCross(
            numHeads, embedSize, encoderEmbedSize, maxSeqLen, maxSeqLen),
        ffn = FeedForward(embedSize),
        ln1 = LayerNorm(embedSize),
        ln2 = LayerNorm(embedSize),
        ln3 = LayerNorm(embedSize);

  /// x_decoder: [T_dec, embedSize]
  /// x_encoder: [T_enc, encoderEmbedSize]
  Tensor forward(Tensor x_decoder, Tensor x_encoder) {
    // 1. Masked AFT Self-Attention + Residual
    // We normalize the whole matrix [T, D] at once
    Tensor x_norm1 = ln1.forward(x_decoder);
    Tensor self_attn_out = selfAttention.forward(x_norm1);
    Tensor x_res1 = x_decoder + self_attn_out;

    // 2. AFT Cross-Attention + Residual
    Tensor x_norm2 = ln2.forward(x_res1);
    Tensor cross_attn_out = crossAttention.forward(x_norm2, x_encoder);
    Tensor x_res2 = x_res1 + cross_attn_out;

    // 3. Feed-Forward + Residual
    Tensor x_norm3 = ln3.forward(x_res2);
    Tensor ffn_out = ffn.forward(x_norm3);
    Tensor out = x_res2 + ffn_out;

    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...selfAttention.parameters(),
        ...crossAttention.parameters(),
        ...ffn.parameters(),
        ...ln1.parameters(),
        ...ln2.parameters(),
        ...ln3.parameters(),
      ];
}

void main() {
  // 1. Hyperparameters (Keeping them identical for comparison)
  const int embedSize = 32;
  const int numHeads = 4;
  const int encoderEmbedSize = 64;
  const int maxSeqLen = 50;
  const double lr = 0.0001; // Lowered to prevent the NaN explosion

  // 2. Initialize Module
  final decoderBlock = TransformerDecoderBlock(
    embedSize,
    numHeads,
    encoderEmbedSize,
    maxSeqLen,
  );

  // 3. Prepare Dummy Tensors
  // Use a fixed seed or simple values if you want to compare outputs exactly
  final xDec = Tensor.random([8, embedSize]);
  final xEnc = Tensor.random([12, encoderEmbedSize]);

  // Create a target tensor (e.g., all 0.5s) to provide a training signal
  final target = Tensor.fill([8, embedSize], 0.5);

  final optimizer = Adam(decoderBlock.parameters(), lr: lr);

  print('--- Comparing TransformerDecoderBlock Outputs ---');
  print('Step | Loss');
  print('-------------------------');

  for (int i = 0; i < 20; i++) {
    optimizer.zeroGrad();

    // 4. Forward Pass
    // Notice: No tracker list needed in this version
    final output = decoderBlock.forward(xDec, xEnc);

    // 5. Compute Loss
    final loss = output.mseLoss(target);

    if (loss.data[0].isNaN) {
      print(
          'Step $i: NaN detected. Try reducing LR further or check initialization.');
      break;
    }

    print(' ${i.toString().padRight(4)}| ${loss.data[0].toStringAsFixed(6)}');

    // 6. Backward Pass
    loss.backward();

    // 7. Gradient Clipping (The "Safety Switch")
    // This prevents the "17347.77" jump you saw earlier
    // for (var p in decoderBlock.parameters()) {
    //   p.grad?.clamp(-1.0, 1.0);
    // }

    // 8. Update Weights
    optimizer.step();

    // Memory Management: If your Tensor class doesn't auto-garbage collect
    // from the GPU, call dispose here on intermediate outputs.
    // output.dispose();
    // loss.dispose();
  }
}
