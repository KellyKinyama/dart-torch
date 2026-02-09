// file: example_aft_full_cycle.dart

import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';

// 1. Simple SGD Optimizer
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
  print("--- AFT-GPT: Full Training & Generation Cycle ---");

  // 1. Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 16; // Smaller for faster training in this example
  const int blockSize = 10;
  const int numLayers = 2;
  const int numHeads = 4;
  const double learningRate = 0.1; // AFT handles higher LR well

  // 2. Vocabulary & Data Setup
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "the": 2,
    "quick": 3,
    "brown": 4,
    "fox": 5,
    ".": 16,
    "<start>": 17,
    "<pad>": 18,
  };
  final Map<int, String> itos = stoi.map((k, v) => MapEntry(v, k));
  final int startId = stoi["<start>"]!;
  final int padId = stoi["<pad>"]!;
  final int endId = stoi["."]!;

  // Training data: "hello world ." and "the quick brown fox ."
  final List<List<int>> trainInputs = [
    [
      startId,
      stoi["hello"]!,
      stoi["world"]!,
      padId,
      padId,
      padId,
      padId,
      padId,
      padId,
      padId
    ],
    [
      startId,
      stoi["the"]!,
      stoi["quick"]!,
      stoi["brown"]!,
      stoi["fox"]!,
      padId,
      padId,
      padId,
      padId,
      padId
    ],
  ];
  final List<List<int>> trainTargets = [
    [
      stoi["hello"]!,
      stoi["world"]!,
      endId,
      padId,
      padId,
      padId,
      padId,
      padId,
      padId,
      padId
    ],
    [
      stoi["the"]!,
      stoi["quick"]!,
      stoi["brown"]!,
      stoi["fox"]!,
      endId,
      padId,
      padId,
      padId,
      padId,
      padId
    ],
  ];

  // 3. Initialize AFT Model
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

  // 4. Training Loop
  print("\n--- Phase 1: Training ---");
  const int epochs = 100;
  for (int epoch = 0; epoch < epochs; epoch++) {
    double epochLoss = 0;

    for (int i = 0; i < trainInputs.length; i++) {
      optimizer.zeroGrad();
      final logits = model.forward(trainInputs[i], dummyEncoder);

      // Cross-Entropy Loss
      Value batchLoss = Value(0.0);
      int active = 0;
      for (int t = 0; t < logits.length; t++) {
        if (trainTargets[i][t] == padId) continue;

        final targetId = trainTargets[i][t];
        final currentLogits = logits[t];

        // Negative Log Likelihood
        final sumExp =
            currentLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
        batchLoss += (sumExp.log() - currentLogits.values[targetId]);
        active++;
      }

      if (active > 0) {
        batchLoss = batchLoss / Value(active.toDouble());
        epochLoss += batchLoss.data;
        batchLoss.backward();
        optimizer.step();
      }
    }
    if ((epoch + 1) % 20 == 0)
      print("Epoch ${epoch + 1}, Loss: ${epochLoss / trainInputs.length}");
  }

  // 5. Phase 2: Generation
  print("\n--- Phase 2: Generation (After Training) ---");
  List<int> gen = [startId];

  for (int i = 0; i < 10; i++) {
    // AFT context windowing
    List<int> input =
        gen.length > blockSize ? gen.sublist(gen.length - blockSize) : gen;

    final logits = model.forward(input, dummyEncoder);
    final probs = logits.last.softmax();

    // Greedy selection
    int nextToken = 0;
    double maxP = -1.0;
    for (int j = 0; j < probs.values.length; j++) {
      if (probs.values[j].data > maxP) {
        maxP = probs.values[j].data;
        nextToken = j;
      }
    }

    gen.add(nextToken);
    print("Step $i: ${gen.map((id) => itos[id] ?? "?").join(' ')}");
    if (nextToken == endId) break;
  }

  print("\nFinal Result: ${gen.map((id) => itos[id] ?? "?").join(' ')}");
}
