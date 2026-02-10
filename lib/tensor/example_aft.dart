import 'dart:math' as math;
import 'dart:typed_data';
import 'tensor.dart';
import 'aft_gpt.dart';

class SGD {
  final List<Tensor> parameters;
  final List<Float32List> velocity; // Stores the "speed" of each weight
  final double learningRate;
  final double momentum = 0.9; // Standard momentum value
  final double clipValue = 1.0;

  SGD(this.parameters, this.learningRate)
      : velocity = parameters.map((p) => Float32List(p.length)).toList();

  void step() {
    for (int i = 0; i < parameters.length; i++) {
      final p = parameters[i];
      final v = velocity[i];

      for (int j = 0; j < p.length; j++) {
        double g = p.grad[j];

        // 1. Clip Gradient
        if (g > clipValue) g = clipValue;
        if (g < -clipValue) g = -clipValue;

        // 2. Update Velocity: v = (momentum * v) - (lr * g)
        v[j] = (momentum * v[j]) - (learningRate * g);

        // 3. Update Weights: p = p + v
        p.data[j] += v[j];
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad.fillRange(0, p.length, 0.0);
    }
  }
}

void main() {
  print("--- Tensor-Engine AFT-GPT Training ---");

  // 1. Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 32;
  const int blockSize = 10;
  const int numLayers = 3;
  const int numHeads = 4;

  // 2. Vocabulary & Data Setup
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

  // 3. Initialize Model & Optimizer
  final model = AFT_GPT(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
  );

  // STABILITY FIX: Lowered learning rate from 0.01 to 0.0005
  // AFT is sensitive to large updates early on.
  const double learningRate = 0.0005;
  final optimizer = SGD(model.parameters(), learningRate);

  // 4. Training Loop
  for (int epoch = 0; epoch < 200; epoch++) {
    // Increased epochs to account for lower LR
    optimizer.zeroGrad();

    final logits = model.forward(inputIds);

    // 5. Vectorized Cross-Entropy Loss
    double lossValue = 0;
    int count = 0;

    for (int t = 0; t < inputIds.length; t++) {
      if (targetIds[t] == stoi["<pad>"]) continue;

      double maxLogit = -double.infinity;
      for (int v = 0; v < vocabSize; v++) {
        if (logits.data[t * vocabSize + v] > maxLogit)
          maxLogit = logits.data[t * vocabSize + v];
      }

      double sumExp = 0;
      for (int v = 0; v < vocabSize; v++) {
        sumExp += math.exp(logits.data[t * vocabSize + v] - maxLogit);
      }

      double logSumExp = maxLogit + math.log(sumExp);
      double targetLogit = logits.data[t * vocabSize + targetIds[t]];

      lossValue += (logSumExp - targetLogit);

      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[t * vocabSize + v] - logSumExp);
        // Standard Softmax Gradient: (p - 1) for target, (p) for others
        logits.grad[t * vocabSize + v] =
            (v == targetIds[t]) ? (prob - 1.0) : prob;
      }
      count++;
    }

    lossValue /= count;
    for (int i = 0; i < logits.grad.length; i++) logits.grad[i] /= count;

    // 6. Backprop and Update
    logits.backward();
    optimizer.step();

    if (epoch % 10 == 0)
      print("Epoch $epoch, Loss: ${lossValue.toStringAsFixed(4)}");
  }

  // 7. Inference (Text Generation)
  print("\nGeneration:");
  List<int> currentSeq = [stoi["<start>"]!];
  for (int i = 0; i < 5; i++) {
    final out = model.forward(currentSeq);

    int nextId = 0;
    double best = -double.infinity;
    int lastOffset = (currentSeq.length - 1) * vocabSize;

    for (int v = 0; v < vocabSize; v++) {
      if (out.data[lastOffset + v] > best) {
        best = out.data[lastOffset + v];
        nextId = v;
      }
    }

    currentSeq.add(nextId);
    if (nextId == stoi["."]) break;
  }
  print(currentSeq.map((id) => itos[id]).join(" "));
}
