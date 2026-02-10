// file: example_aft_4gb_cycle.dart

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
  print("--- AFT-GPT: Optimized for 4GB RAM ---");

  // 1. Scaled Hyperparameters for 4GB
  // embedSize: 48-64 is the limit for a smooth 4GB experience.
  // numLayers: 3 keeps the backward chain manageable.
  // blockSize: 32 balances context with the AFT bias matrix size.
  const int vocabSize = 50;
  const int embedSize = 48;
  const int blockSize = 32;
  const int numLayers = 3;
  const int numHeads = 4;
  const double learningRate = 0.05;

  // 2. Model Initialization
  final model = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );

  final optimizer = SGD(model.parameters(), learningRate);

  // Minimal dummy encoder output to save memory
  final List<ValueVector> dummyEncoder = [
    ValueVector(List.filled(embedSize, Value(0.0)))
  ];

  print("Model initialized: ${model.parameters().length} Parameters.");
  print("Peak RAM Target: ~2.5 - 3.5 GB");

  // 3. Training Loop
  for (int epoch = 0; epoch < 100; epoch++) {
    // Explicitly scope the training step logic
    double currentLoss =
        trainStep(model, optimizer, vocabSize, blockSize, dummyEncoder);

    if (epoch % 10 == 0) {
      print("Epoch $epoch | Loss: ${currentLoss.toStringAsFixed(4)}");
    }

    // Hint to Dart: You can run GC now if needed
    // (In native compiled mode, this is more effective)
  }
}

/// Wrapping the logic in a function ensures that temporary 'Value' objects
/// and the computation graph fall out of scope as soon as the function returns.
double trainStep(TransformerDecoder model, SGD optimizer, int vocabSize,
    int blockSize, List<ValueVector> dummyEncoder) {
  optimizer.zeroGrad();

  // Generate synthetic training data
  List<int> input = List.generate(blockSize, (i) => i % vocabSize);
  List<int> target = List.generate(blockSize, (i) => (i + 1) % vocabSize);

  // FORWARD PASS
  final logits = model.forward(input, dummyEncoder);

  // LOSS CALCULATION
  Value totalLoss = Value(0.0);
  for (int t = 0; t < logits.length; t++) {
    final targetId = target[t];

    // Stable LogSumExp
    final sumExp = logits[t].values.map((v) => v.exp()).reduce((a, b) => a + b);
    totalLoss += (sumExp.log() - logits[t].values[targetId]);
  }

  Value finalLoss = totalLoss / Value(blockSize.toDouble());

  // BACKWARD PASS (The 4GB Peak)
  finalLoss.backward();

  // UPDATE
  optimizer.step();

  return finalLoss.data;
}
