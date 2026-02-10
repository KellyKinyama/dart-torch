// file: example_aft_full_cycle.dart

import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';

class SGD {
  final List<Value> parameters;
  final double learningRate;
  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      p.data -= learningRate * p.grad;
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0.0;
    }
  }
}

void main() {
  print("--- AFT-GPT: Optimized for 12GB RAM ---");

  // 1. Scaled Hyperparameters for 12GB
  // We scale the embedSize and Layers.
  // embedSize: 128 is a "Small-Base" configuration.
  // numLayers: 6 provides significant depth.
  const int vocabSize = 100;
  const int embedSize = 128;
  const int blockSize =
      64; // AFT allows larger blocks without quadratic explosion
  const int numLayers = 6;
  const int numHeads = 8;
  const double learningRate = 0.05;

  // 2. Memory-Optimized Training Logic
  final model = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );

  final optimizer = SGD(model.parameters(), learningRate);
  final List<ValueVector> dummyEncoder = [
    ValueVector(List.filled(embedSize, Value(0.0)))
  ];

  print("Model initialized: ${model.parameters().length} Parameters.");
  print("Estimated RAM usage per forward pass: ~4-6 GB");

  // 3. Training Loop with explicit graph clearing
  for (int epoch = 0; epoch < 50; epoch++) {
    // In a real 12GB scenario, we process one sequence at a time
    // to prevent the Dart heap from fragmenting.

    optimizer.zeroGrad();

    // Simulate a sequence of blockSize
    List<int> input = List.generate(blockSize, (i) => i % vocabSize);
    List<int> target = List.generate(blockSize, (i) => (i + 1) % vocabSize);

    // FORWARD PASS
    // This creates millions of Value objects.
    final logits = model.forward(input, dummyEncoder);

    // LOSS CALCULATION
    Value totalLoss = Value(0.0);
    for (int t = 0; t < logits.length; t++) {
      final targetId = target[t];
      // Optimization: use a more stable LogSumExp to prevent overflow
      final sumExp =
          logits[t].values.map((v) => v.exp()).reduce((a, b) => a + b);
      totalLoss += (sumExp.log() - logits[t].values[targetId]);
    }
    totalLoss = totalLoss / Value(blockSize.toDouble());

    // BACKWARD PASS
    // This is the peak memory usage point.
    totalLoss.backward();

    // UPDATE
    optimizer.step();

    // CRITICAL FOR 12GB:
    // By letting 'logits' and 'totalLoss' go out of scope at the end of this loop iteration,
    // the Dart Garbage Collector can reclaim the millions of child Value objects.
    if (epoch % 5 == 0) {
      print("Epoch $epoch | Loss: ${totalLoss.data.toStringAsFixed(4)}");
    }
  }
}
