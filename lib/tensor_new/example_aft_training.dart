import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'aft_transformer_decoder.dart';

class SGD {
  final List<Tensor> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      for (int i = 0; i < p.data.length; i++) {
        p.data[i] -= learningRate * p.grad[i];
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad.fillRange(0, p.grad.length, 0.0);
    }
  }
}

void main() {
  print("--- Tensor-Based AFT Training (Seq2Seq Context) ---");

  // 1. Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 32;
  const int blockSize = 10;
  const double learningRate = 0.01;

  // 2. Vocabulary & Data (Simplified for brevity)
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    ".": 16,
    "<start>": 17,
    "<pad>": 18
  };
  final itos = stoi.map((k, v) => MapEntry(v, k));
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;

  // Dummy Dataset: [<start>, hello, world, .]
  final List<int> sequence = [startTokenId, 0, 1, 16];
  final List<int> trainInput = [startTokenId, 0, 1]; // X
  final List<int> trainTarget = [0, 1, 16]; // Y (Shifted)

  // 3. Model & Optimizer
  final model = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    encoderEmbedSize: embedSize,
  );
  final optimizer = SGD(model.parameters(), learningRate);

  // Dummy Encoder context [1, embedSize]
  final encoderOutput = Tensor.zeros([1, embedSize]);

  // 4. Training Loop
  print("\nStarting Training...");
  for (int epoch = 0; epoch < 500; epoch++) {
    optimizer.zeroGrad();

    // Forward: [T, vocabSize]
    Tensor logits = model.forward(trainInput, encoderOutput);

    // 5. Tensor-based Cross Entropy Loss
    // loss = -sum(log_softmax(logits)[target]) / T
    double lossValue = 0.0;

    // We compute gradients manually for the loss layer to feed the graph
    // (In a full engine, you'd have a CrossEntropy node)
    for (int t = 0; t < trainInput.length; t++) {
      int targetId = trainTarget[t];

      // Log-Sum-Exp trick for stability
      double maxLogit = logits.getRow(t).data.reduce((a, b) => a > b ? a : b);
      double sumExp = 0;
      for (var val in logits.getRow(t).data) {
        sumExp += math.exp(val - maxLogit);
      }
      double logProb =
          logits.data[t * vocabSize + targetId] - maxLogit - math.log(sumExp);

      lossValue -= logProb;

      // Set gradients for backprop: (softmax - 1 at target)
      for (int v = 0; v < vocabSize; v++) {
        double p = math.exp(logits.data[t * vocabSize + v] - maxLogit) / sumExp;
        logits.grad[t * vocabSize + v] += (v == targetId) ? (p - 1.0) : p;
      }
    }

    lossValue /= trainInput.length;

    // Backward & Update
    logits.backward();
    optimizer.step();

    if ((epoch + 1) % 50 == 0) {
      print("Epoch ${epoch + 1}, Loss: ${lossValue.toStringAsFixed(4)}");
    }
  }

  // 6. Test Inference
  print("\nInference after training:");
  List<int> gen = [startTokenId];
  for (int i = 0; i < 5; i++) {
    Tensor out = model.forward(gen, encoderOutput);
    int next = _argmax(out.getRow(gen.length - 1));
    gen.add(next);
    if (next == 16) break;
  }
  print("Result: ${gen.map((id) => itos[id] ?? '??').join(' ')}");
}

int _argmax(Tensor t) {
  double maxV = double.negativeInfinity;
  int idx = 0;
  for (int i = 0; i < t.data.length; i++) {
    if (t.data[i] > maxV) {
      maxV = t.data[i];
      idx = i;
    }
  }
  return idx;
}
