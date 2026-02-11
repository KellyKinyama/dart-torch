import 'linear.dart';
import 'module.dart';
import 'tensor.dart';
import 'dart:math' as math;

class MultiHeadAFTCross extends Module {
  final Linear qProj;
  final Linear kProj;
  final Linear vProj;
  final Linear outProj;
  final Tensor posBias;
  final int numHeads;
  final int headSize;

  MultiHeadAFTCross(
      int numHeads, int embedSize, int encoderEmbedSize, int maxSeqLen)
      : headSize = embedSize ~/ numHeads,
        numHeads = numHeads,
        qProj = Linear(embedSize, embedSize),
        kProj = Linear(encoderEmbedSize, embedSize),
        vProj = Linear(encoderEmbedSize, embedSize),
        outProj = Linear(embedSize, embedSize),
        // Initialize with small constant to keep initial gradients stable
        posBias = Tensor.fill([maxSeqLen, maxSeqLen], 0.01);

  /// x_dec: [T_dec, EmbedSize] (The "Query" source from Decoder)
  /// x_enc: [T_enc, EncoderEmbedSize] (The "Key/Value" source from Encoder)
  Tensor forward(Tensor x_dec, Tensor x_enc) {
    final T_dec = x_dec.shape[0];
    final T_enc = x_enc.shape[0];

    // 1. Projections
    // Q uses Sigmoid gating as per AFT paper
    final Q = qProj.forward(x_dec).sigmoid();

    // 2. K/V Stability
    // We clamp the input to Exp to prevent Infinity/NaN
    final rawK = kProj.forward(x_enc);
    final K_exp = _clamp(rawK, -80.0, 80.0).exp();
    final V = vProj.forward(x_enc);

    // 3. Vectorized AFT-Cross Logic
    // We slice the position bias to match the current Dec/Enc sequence lengths
    final currentW = posBias.slice2D(T_dec, T_enc);

    // Numerator: (currentW matmul (K_exp * V))
    final expKV = K_exp * V;
    final num = currentW.matmul(expKV); // Result: [T_dec, EmbedSize]

    // Denominator: (currentW matmul K_exp)
    final den = currentW.matmul(K_exp); // Result: [T_dec, EmbedSize]

    // 4. Combine and Project
    // Stability Fix: Epsilon prevents division by zero if K is very small
    // Logic: Q * (Numerator / Denominator)
    final aftOut = Q * (num / (den + 1e-9));

    return outProj.forward(aftOut);
  }

  /// Internal helper to prevent exploding gradients/values
  Tensor _clamp(Tensor x, double min, double max) {
    final out = Tensor(x.shape, children: {x});
    for (int i = 0; i < x.length; i++) {
      out.data[i] = x.data[i].clamp(min, max);
    }
    out.onBackward = () {
      for (int i = 0; i < x.length; i++) {
        if (x.data[i] >= min && x.data[i] <= max) {
          x.grad[i] += out.grad[i];
        }
      }
    };
    return out;
  }

  @override
  List<Tensor> parameters() => [
        ...qProj.parameters(),
        ...kProj.parameters(),
        ...vProj.parameters(),
        ...outProj.parameters(),
        posBias
      ];
}

void main() {
  // 1. Hyperparameters
  const int numHeads = 4;
  const int decoderEmbed = 32; // The dimension the decoder is working with
  const int encoderEmbed = 64; // The dimension of the encoder's output
  const int maxSeqLen = 50; // Max capacity for position bias
  const double lr = 0.01;

  // 2. Initialize Module
  // Note: encoderEmbedSize can be different from embedSize
  final aftCross =
      MultiHeadAFTCross(numHeads, decoderEmbed, encoderEmbed, maxSeqLen);

  // 3. Prepare Dummy Tensors
  // Decoder sequence (e.g., words already translated)
  final xDec = Tensor.random([5, decoderEmbed]);

  // Encoder sequence (e.g., the full source sentence being translated)
  final xEnc = Tensor.random([12, encoderEmbed]);

  // Target output for this step [T_dec, decoderEmbed]
  final target = Tensor.fill([5, decoderEmbed], 0.1);

  print('--- MultiHeadAFTCross Training Step ---');

  // 4. Forward Pass
  // The decoder queries the encoder
  final output = aftCross.forward(xDec, xEnc);
  print('Output shape: ${output.shape}'); // Expected: [5, 32]

  // 5. Compute Loss
  final loss = output.mseLoss(target);
  print('Initial Cross-Attention Loss: ${loss.data[0].toStringAsFixed(6)}');

  // 6. Backward Pass
  // This backpropagates through the decoder projections AND encoder projections
  loss.backward();

  // 7. Manual SGD Update
  for (var p in aftCross.parameters()) {
    for (int i = 0; i < p.length; i++) {
      p.data[i] -= lr * p.grad[i];
    }
    p.zeroGrad();
  }

  // 8. Verify
  final nextLoss = aftCross.forward(xDec, xEnc).mseLoss(target);
  print(
      'Loss after Cross-Attention update: ${nextLoss.data[0].toStringAsFixed(6)}');
}
