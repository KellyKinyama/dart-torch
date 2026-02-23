import 'dart:math' as math;
import 'dart:typed_data';
import 'optimizer.dart';
import 'tensor.dart';
import 'aft_gpt.dart';

// class SGD {
//   final List<Tensor> parameters;
//   final List<Float32List> velocity;
//   final double learningRate;
//   final double momentum;
//   final double clipValue = 1.0; // Tightened clipping for AFT stability
//   final double weightDecay = 0.0001;

//   SGD(this.parameters, this.learningRate, {this.momentum = 0.0})
//       : velocity = parameters.map((p) => Float32List(p.length)).toList();

//   void step() {
//     // 1. Global Norm Clipping
//     double globalNorm = 0.0;
//     for (var p in parameters) {
//       for (var g in p.grad) globalNorm += g * g;
//     }
//     globalNorm = math.sqrt(globalNorm);

//     // Scaling factor to keep gradients sane
//     double scale = (globalNorm > clipValue) ? (clipValue / globalNorm) : 1.0;

//     for (int i = 0; i < parameters.length; i++) {
//       final p = parameters[i];
//       final v = velocity[i];

//       for (int j = 0; j < p.length; j++) {
//         // 2. Weight Decay + Scaled Gradient
//         double g = (p.grad[j] * scale) + (weightDecay * p.data[j]);

//         // 3. Momentum Math (Now correctly using the class variable)
//         v[j] = (momentum * v[j]) - (learningRate * g);
//         p.data[j] += v[j];

//         // 4. Safety Rail
//         if (p.data[j].isNaN || p.data[j].isInfinite) p.data[j] = 0.0;
//       }
//     }
//   }

//   void zeroGrad() {
//     for (final p in parameters) {
//       p.grad.fillRange(0, p.length, 0.0);
//     }
//   }
// }

void main() {
  print("--- Stable Tensor-Engine AFT-GPT Training ---");

  const int vocabSize = 5;
  const int embedSize = 32; // Smaller is often more stable for toy examples
  const int blockSize = 16;
  const int numLayers = 2; // Start with 1 layer to ensure convergence

  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    ".": 2,
    "<start>": 3,
    "<pad>": 4
  };
  final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));

  final List<int> inputIds = [3, 0, 1, 4, 4, 4, 4, 4, 4, 4];
  final List<int> targetIds = [0, 1, 2, 4, 4, 4, 4, 4, 4, 4];

  final model = AFT_GPT(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: 4,
  );

  // Use a conservative Learning Rate without momentum first
  const double learningRate = 0.01;
  final optimizer = SGD(
    model.parameters(), lr: learningRate, //momentum: 0.0
  );

  print('Starting training...');
  for (int epoch = 0; epoch <= 300; epoch++) {
    optimizer.zeroGrad();
    final logits = model.forward(inputIds);

    double lossValue = 0;
    int activeTokens = 0;

    for (int t = 0; t < inputIds.length; t++) {
      if (targetIds[t] == stoi["<pad>"]) continue;
      activeTokens++;

      int rowOffset = t * vocabSize;

      // Log-Sum-Exp trick
      double maxLogit = -double.infinity;
      for (int v = 0; v < vocabSize; v++) {
        if (logits.data[rowOffset + v] > maxLogit) {
          maxLogit = logits.data[rowOffset + v];
        }
      }

      double sumExp = 0;
      for (int v = 0; v < vocabSize; v++) {
        sumExp += math.exp(logits.data[rowOffset + v] - maxLogit);
      }

      double logSumExp = maxLogit + math.log(sumExp);
      lossValue += (logSumExp - logits.data[rowOffset + targetIds[t]]);

      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[rowOffset + v] - logSumExp);
        logits.grad[rowOffset + v] = (v == targetIds[t]) ? (prob - 1.0) : prob;
      }
    }

    lossValue /= activeTokens;

    // Normalize gradients by the number of samples in the batch
    for (int i = 0; i < logits.grad.length; i++) {
      logits.grad[i] /= activeTokens;
    }

    logits.backward();
    optimizer.step();

    // if (epoch % 50 == 0) {
    print("Epoch $epoch | Loss: ${lossValue.toStringAsFixed(4)}");
    if (lossValue.isNaN) break;
    // }
  }

  // 3. Inference
  print("\nInference Result:");
  List<int> currentSeq = [stoi["<start>"]!];
  for (int i = 0; i < 5; i++) {
    final out = model.forward(currentSeq);
    int lastOffset = (currentSeq.length - 1) * vocabSize;

    int nextId = 0;
    double best = -double.infinity;
    for (int v = 0; v < vocabSize; v++) {
      if (out.data[lastOffset + v] > best) {
        best = out.data[lastOffset + v];
        nextId = v;
      }
    }
    currentSeq.add(nextId);
    if (nextId == stoi["."]) break;
  }
  print("Output: ${currentSeq.map((id) => itos[id] ?? "??").join(" ")}");
}
