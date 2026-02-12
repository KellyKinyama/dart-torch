import 'dart:math' as math;
import '../tensor/tensor.dart';
import 'aft_transformer_decoder.dart';

// 1. Efficient Tensor-based SGD Optimizer
class SGD {
  final List<Tensor> parameters;
  final double lr;

  SGD(this.parameters, this.lr);

  void step() {
    for (var p in parameters) {
      for (int i = 0; i < p.data.length; i++) {
        p.data[i] -= lr * p.grad[i];
      }
    }
  }

  void zeroGrad() {
    for (var p in parameters) {
      p.grad.fillRange(0, p.grad.length, 0.0);
    }
  }
}

// 2. Cross Entropy Loss Node with explicit onBackward trigger
Tensor crossEntropy(Tensor logits, List<int> targets, int vocabSize) {
  int numTokens = targets.length;
  double totalLoss = 0;

  // Forward pass
  for (int t = 0; t < numTokens; t++) {
    int target = targets[t];
    int offset = t * vocabSize;

    double maxL = -double.infinity;
    for (int v = 0; v < vocabSize; v++) {
      if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];
    }

    double sumExp = 0;
    for (int v = 0; v < vocabSize; v++) {
      sumExp += math.exp(logits.data[offset + v] - maxL);
    }
    double logSumExp = maxL + math.log(sumExp + 1e-12);
    totalLoss += (logSumExp - logits.data[offset + target]);
  }

  // Create the loss node linked to logits
  final loss = Tensor([1], children: {logits});
  loss.data[0] = totalLoss / numTokens;

  // Define how gradients flow back through the Logits
  loss.onBackward = () {
    double gradFromLoss = 1.0 / numTokens;
    for (int t = 0; t < numTokens; t++) {
      int target = targets[t];
      int offset = t * vocabSize;

      double maxL = -double.infinity;
      for (int v = 0; v < vocabSize; v++) {
        if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];
      }

      double sumExp = 0;
      for (int v = 0; v < vocabSize; v++) {
        sumExp += math.exp(logits.data[offset + v] - maxL);
      }

      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[offset + v] - maxL) / sumExp;
        double targetSignal = (v == target) ? 1.0 : 0.0;
        // Accumulate gradients into the logits tensor
        logits.grad[offset + v] += (prob - targetSignal) * gradFromLoss;
      }
    }
  };

  return loss;
}

void main() {
  print("--- Tensor-Based GPT Diagnostic Training ---");

  const vocabSize = 40;
  const embedSize = 32;
  const blockSize = 15;
  const double learningRate = 0.05;

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
  );

  final optimizer = SGD(gpt.parameters(), learningRate);
  final dummyEncoder = Tensor.zeros([1, embedSize]);

  // Input Mapping
  List<int> inputIds = [7, 0, 1];
  List<int> targetIds = [0, 1, 6];

  print("\n--- Starting Training ---");
  for (int epoch = 0; epoch <= 100; epoch++) {
    optimizer.zeroGrad();

    // 1. Forward
    final logits = gpt.forward(inputIds, dummyEncoder);

    // 2. Loss Tensor
    final loss = crossEntropy(logits, targetIds, vocabSize);

    // 3. Backward Pass (Triggers onBackward for all nodes)
    loss.backward();

    // 4. Trace progress
    if (epoch % 20 == 0) {
      var lastLayer = gpt.parameters().last;
      double gradMagnitude = lastLayer.grad.fold(0, (a, b) => a + b.abs());
      print(
          "Epoch $epoch | Loss: ${loss.data[0].toStringAsFixed(6)} | Grad: $gradMagnitude");
    }

    optimizer.step();
  }

  print("\n--- Diagnostic Results ---");
  final finalLogits = gpt.forward(inputIds, dummyEncoder);
  for (int t = 0; t < inputIds.length; t++) {
    int predictedId = _argmaxRow(finalLogits, t, vocabSize);
    print("In: ${inputIds[t]} | Out: $predictedId (Target: ${targetIds[t]})");
  }
}

int _argmaxRow(Tensor logits, int row, int vocabSize) {
  int offset = row * vocabSize;
  int maxIdx = 0;
  double maxV = -double.infinity;
  for (int v = 0; v < vocabSize; v++) {
    if (logits.data[offset + v] > maxV) {
      maxV = logits.data[offset + v];
      maxIdx = v;
    }
  }
  return maxIdx;
}
