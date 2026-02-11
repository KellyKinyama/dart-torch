import 'linear.dart';
import 'module.dart';
import 'tensor.dart';
import 'dart:math' as math;

class MultiHeadAFT extends Module {
  final Linear qkvProj;
  final Linear outProj;
  final Tensor posBias;
  final int embedSize;
  final bool masked; // Restored field

  MultiHeadAFT(int numHeads, this.embedSize, int maxSeqLen,
      {this.masked = false})
      : qkvProj = Linear(embedSize, 3 * embedSize),
        outProj = Linear(embedSize, embedSize),
        posBias = Tensor.fill([maxSeqLen, maxSeqLen], 0.01);

  @override
  List<Tensor> parameters() =>
      [...qkvProj.parameters(), ...outProj.parameters(), posBias];

  @override
  Tensor forward(Tensor x) {
    final T = x.shape[0];
    final qkv = qkvProj.forward(x);

    final Q = _extract(qkv, 0, embedSize).sigmoid();
    final K = _extract(qkv, embedSize, 2 * embedSize);
    final V = _extract(qkv, 2 * embedSize, 3 * embedSize);
    final w = posBias.slice2D(T, T);

    final out = Tensor([T, embedSize], children: {Q, K, V, w});

    final List<double> denominators = List.filled(T * embedSize, 0.0);
    final List<double> numerators = List.filled(T * embedSize, 0.0);

    for (int t = 0; t < T; t++) {
      for (int tp = 0; tp < T; tp++) {
        // CAUSAL MASKING LOGIC:
        // If masked is true, we only allow tp <= t (past and present)
        if (masked && tp > t) continue;

        for (int e = 0; e < embedSize; e++) {
          double bias = w.data[t * T + tp];
          double val =
              math.exp((K.data[tp * embedSize + e] + bias).clamp(-80, 80));

          numerators[t * embedSize + e] += val * V.data[tp * embedSize + e];
          denominators[t * embedSize + e] += val;
        }
      }

      for (int e = 0; e < embedSize; e++) {
        double context = numerators[t * embedSize + e] /
            (denominators[t * embedSize + e] + 1e-9);
        out.data[t * embedSize + e] = Q.data[t * embedSize + e] * context;
      }
    }

    out.onBackward = () {
      for (int t = 0; t < T; t++) {
        for (int e = 0; e < embedSize; e++) {
          int idx = t * embedSize + e;
          double dOut = out.grad[idx];
          double qVal = Q.data[idx];
          double numVal = numerators[idx];
          double denVal = denominators[idx] + 1e-9;
          double contextVal = numVal / denVal;

          Q.grad[idx] += dOut * contextVal;

          double dNum = dOut * qVal / denVal;
          double dDen = dOut * qVal * (-numVal / (denVal * denVal));

          for (int tp = 0; tp < T; tp++) {
            // Respect the mask in backward pass too
            if (masked && tp > t) continue;

            double bias = w.data[t * T + tp];
            double kVal = K.data[tp * embedSize + e];
            double expVal = math.exp((kVal + bias).clamp(-80, 80));
            double vVal = V.data[tp * embedSize + e];

            double dExp = (dNum * vVal) + dDen;
            double localGrad = dExp * expVal;

            K.grad[tp * embedSize + e] += localGrad;
            w.grad[t * T + tp] += localGrad;
            V.grad[tp * embedSize + e] += dNum * expVal;
          }
        }
      }
    };

    return outProj.forward(out);
  }

  Tensor _extract(Tensor src, int start, int end) {
    final T = src.shape[0];
    final E = end - start;
    final res = Tensor([T, E], children: {src});
    for (int t = 0; t < T; t++) {
      for (int e = 0; e < E; e++) {
        res.data[t * E + e] = src.data[t * (src.shape[1]) + start + e];
      }
    }
    res.onBackward = () {
      for (int t = 0; t < T; t++) {
        for (int e = 0; e < E; e++) {
          src.grad[t * src.shape[1] + start + e] += res.grad[t * E + e];
        }
      }
    };
    return res;
  }
}

void main() {
  // 1. Hyperparameters
  const int numHeads =
      4; // Note: The logic provided processes embedSize directly
  const int embedSize = 16; // Total embedding dimension
  const int maxSeqLen = 20; // Capacity of the position bias matrix
  const int currentT = 8; // Actual sequence length for this batch
  const double lr = 0.01; // Learning rate

  // 2. Initialize Module (with Causal Masking enabled)
  // Masked: true is typical for Autoregressive tasks (like GPT)
  final aft = MultiHeadAFT(numHeads, embedSize, maxSeqLen, masked: true);

  // 3. Create Dummy Data [T, Dim]
  final input = Tensor.random([currentT, embedSize]);

  // Target: We want the model to predict a specific pattern (e.g., all 0.5s)
  final target = Tensor.fill([currentT, embedSize], 0.5);

  print('--- MultiHeadAFT Training Step (Causal) ---');

  // 4. Forward Pass
  // The module internally handles the slicing of posBias and the causal loop
  final output = aft.forward(input);
  print('Output shape: ${output.shape}');

  // 5. Compute Loss
  final loss = output.mseLoss(target);
  print('Initial Loss: ${loss.data[0].toStringAsFixed(6)}');

  // 6. Backward Pass
  // This triggers the custom onBackward blocks for extraction,
  // the AFT logic loop, and the linear projections.
  loss.backward();

  // 7. Optimizer Step (Manual SGD)
  final params = aft.parameters();
  int paramCount = 0;
  for (var p in params) {
    for (int i = 0; i < p.length; i++) {
      p.data[i] -= lr * p.grad[i];
    }
    p.zeroGrad(); // Essential: Clear gradients after the update
    paramCount++;
  }
  print('Updated $paramCount parameter tensors.');

  // 8. Verify Progress
  final nextOutput = aft.forward(input);
  final nextLoss = nextOutput.mseLoss(target);
  print('Loss after 1 step: ${nextLoss.data[0].toStringAsFixed(6)}');

  if (nextLoss.data[0] < loss.data[0]) {
    print('Success: Gradients flowed and loss decreased!');
  }
}
