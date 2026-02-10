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
