import 'dart:math' as math;
import '../tensor/tensor.dart';
import 'transformer_decoder.dart';

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

// 2. Helper function to create a Loss Tensor that preserves the Autograd chain
Tensor crossEntropy(Tensor logits, List<int> targets, int vocabSize) {
  int numTokens = targets.length;
  double totalLoss = 0;

  // Forward pass: Calculate Cross Entropy
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

  // Create a new Tensor for the loss value
  final loss = Tensor([1], children: {logits});
  loss.data[0] = totalLoss / numTokens;

  // Backward pass: Seed the logits.grad correctly
  loss.onBackward = () {
    double gradFromLoss = 1.0 / numTokens; // Mean reduction
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

      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[offset + v] - logSumExp);
        double targetSignal = (v == target) ? 1.0 : 0.0;
        // This line adds to the existing chain instead of breaking it
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
  const double learningRate = 0.05; // Slightly higher for faster convergence

  final gpt = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
  );

  final optimizer = SGD(gpt.parameters(), learningRate);
  final dummyEncoder = Tensor.zeros([1, embedSize]);

  // Input: [<start>, hello, world] | Target: [hello, world, .]
  List<int> inputIds = [7, 0, 1];
  List<int> targetIds = [0, 1, 6];

  print("\n--- Starting Training ---");
  for (int epoch = 0; epoch <= 100; epoch++) {
    optimizer.zeroGrad();

    // 1. Forward Pass
    final logits = gpt.forward(inputIds, dummyEncoder);

    // 2. Cross Entropy Loss (Now returns a proper Tensor node)
    final loss = crossEntropy(logits, targetIds, vocabSize);

    // 3. Topology check on first epoch
    if (epoch == 0) {
      print("--- Graph Topology Check ---");
      final visited = <Tensor>{};
      void check(Tensor t) {
        if (visited.add(t)) {
          for (final p in t.parents) check(p);
        }
      }

      check(loss);
      print("Nodes in graph: ${visited.length}");
      bool reachable = visited.contains(gpt.tokenEmbeddingTable);
      print("Embedding Table reachable from Loss? $reachable");
      print("----------------------------\n");
    }

    // 4. Backward Pass (from the loss node)
    loss.backward();

    // 5. Training trace
    if (epoch % 20 == 0) {
      var params = gpt.parameters();
      var lastW = params.last;
      double lastWGrad = lastW.grad.fold(0, (a, b) => a + b.abs());

      print(
          "Epoch $epoch | Loss: ${loss.data[0].toStringAsFixed(6)} | Last Layer Grad: $lastWGrad");
    }

    optimizer.step();
  }

  print("\n--- Inference Test ---");
  final testEncoder = Tensor.zeros([1, embedSize]); // Keep encoder neutral
  final finalLogits = gpt.forward(inputIds, testEncoder);

  for (int t = 0; t < inputIds.length; t++) {
    int offset = t * vocabSize;
    int predictedId = 0;
    double maxProb = -double.infinity;

    for (int v = 0; v < vocabSize; v++) {
      if (finalLogits.data[offset + v] > maxProb) {
        maxProb = finalLogits.data[offset + v];
        predictedId = v;
      }
    }
    print(
        "Input ID: ${inputIds[t]} | Predicted Next ID: $predictedId (Target: ${targetIds[t]})");
  }
}
