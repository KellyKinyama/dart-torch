import 'dart:math' as math;

import 'package:dart_torch/tensor_new/adam.dart';

import '../tensor/tensor.dart';
import 'aft_transformer_decoder.dart';
import 'example_aft_main.dart';

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
  final optimizer = Adam(model.parameters(), lr: learningRate);

  // Dummy Encoder context [1, embedSize]
  final encoderOutput = Tensor.zeros([1, embedSize]);

  // 4. Training Loop
  print("\nStarting Training...");
  for (int epoch = 0; epoch < 5000; epoch++) {
    optimizer.zeroGrad();

    // Forward: [T, vocabSize]
    Tensor logits = model.forward(trainInput, encoderOutput);

    // 5. Tensor-based Cross Entropy Loss
    // loss = -sum(log_softmax(logits)[target]) / T
    // double lossValue = 0.0;

    final loss = crossEntropy(logits, trainTarget, vocabSize);

    // lossValue /= trainInput.length;

    // Backward & Update
    loss.backward();
    optimizer.step();

    if ((epoch + 1) % 50 == 0) {
      print("Epoch ${epoch + 1}, Loss: ${loss.data[0].toStringAsFixed(4)}");
    }
  }

  // 6. Test Inference
  print("\nInference after training:");
  List<int> gen = [
    startTokenId,
  ];
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
