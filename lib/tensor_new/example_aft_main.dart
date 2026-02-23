import 'dart:math' as math;

import '../tensor/tensor.dart';
import 'adam.dart';
// import 'example_aft_full_cycle.dart';
import 'aft_transformer_decoder.dart';

// --- TEMPERATURE SAMPLING ---
int sample(Tensor logits, int row, int vocabSize, double temperature) {
  int offset = row * vocabSize;
  List<double> probs = [];
  double sumExp = 0;

  // Apply Temperature and Softmax
  double maxL = -double.infinity;
  for (int v = 0; v < vocabSize; v++) {
    double val = logits.data[offset + v] / temperature;
    if (val > maxL) maxL = val;
  }

  for (int v = 0; v < vocabSize; v++) {
    double p = math.exp((logits.data[offset + v] / temperature) - maxL);
    probs.add(p);
    sumExp += p;
  }

  // Random Selection
  double r = math.Random().nextDouble() * sumExp;
  double cumulative = 0;
  for (int i = 0; i < vocabSize; i++) {
    cumulative += probs[i];
    if (r <= cumulative) return i;
  }
  return vocabSize - 1;
}

void generate(TransformerDecoder model, List<int> gen, int endId,
    Map<int, String> itos, int vocabSize, int blockSize, Tensor dummyEnc) {
  print("Seed: ${gen.map((id) => itos[id]).join(' ')}");

  for (int i = 0; i < 10; i++) {
    List<int> context =
        gen.length > blockSize ? gen.sublist(gen.length - blockSize) : gen;
    final logits = model.forward(context, dummyEnc);

    // We use temperature 0.1 (very confident) to avoid random junk
    int nextId = sample(logits, context.length - 1, vocabSize, 0.1);

    gen.add(nextId);
    print("  Next -> ${itos[nextId] ?? '?'}");
    if (nextId == endId) break;
  }
  print("Result: ${gen.map((id) => itos[id] ?? '?').join(' ')}");
}

// --- THE LOSS FUNCTION (Autograd Node) ---
Tensor crossEntropy(Tensor logits, List<int> targets, int vocabSize) {
  int numTokens = targets.length;
  double totalLoss = 0;

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
    totalLoss +=
        (maxL + math.log(sumExp + 1e-12) - logits.data[offset + target]);
  }

  final loss = Tensor([1], children: {logits});
  loss.data[0] = totalLoss / numTokens;

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
        logits.grad[offset + v] +=
            (prob - ((v == target) ? 1.0 : 0.0)) * gradFromLoss;
      }
    }
  };
  return loss;
}

void main() {
  print("--- Training Medium AFT-GPT ---");

  // 1. Scaled-down but Robust Hyperparameters
  const int vocabSize = 25;
  const int embedSize = 64; // Fixed: Mid-point between 32 and 128
  const int blockSize = 16; // Sequence length
  const int numLayers = 4; // Depth
  const int numHeads = 4; // Head Dimension = 64 / 4 = 16

  final stoi = {
    "hello": 0,
    "world": 1,
    "the": 2,
    "quick": 3,
    "brown": 4,
    "fox": 5,
    ".": 6,
    "<start>": 7,
    "jumps": 8,
    "over": 9
  };
  final itos = stoi.map((k, v) => MapEntry(v, k));

  // 2. Model Initialization
  // Ensure your constructor uses the constants below!
  const int bigSize = 64; // Define once, use everywhere

  final gpt = TransformerDecoder(
    vocabSize: 25,
    embedSize: bigSize,
    encoderEmbedSize: bigSize, // Must match!
    numLayers: 1,
    numHeads: 2,
    blockSize: 16,
  );

  // 3. Adam Optimizer (Use a slightly higher LR for the smaller model)
  final optimizer = Adam(gpt.parameters(), lr: 0.001);
  final dummyEnc = Tensor.zeros([1, embedSize]);

  final dataset = [
    [7, 0, 1, 6], // <start> hello world .
    [7, 2, 3, 4, 5, 6], // <start> the quick brown fox .
    [
      7,
      2,
      3,
      4,
      5,
      8,
      9,
      0,
      6
    ] // <start> the quick brown fox jumps over hello .
  ];

  // 4. Training Loop
  print("Total Parameter Tensors: ${gpt.parameters().length}");

  for (int epoch = 0; epoch <= 1000; epoch++) {
    double totalLoss = 0;
    for (var seq in dataset) {
      optimizer.zeroGrad();
      final x = seq.sublist(0, seq.length - 1);
      final y = seq.sublist(1);

      final logits = gpt.forward(x, dummyEnc);
      final loss = crossEntropy(logits, y, vocabSize);
      loss.backward();
      optimizer.step();
      totalLoss += loss.data[0];
    }

    if (epoch % 50 == 0) {
      print(
          "Epoch $epoch | Loss: ${(totalLoss / dataset.length).toStringAsFixed(6)}");
    }
  }

  // 5. Test Generation
  print("\n--- Testing Model Memory ---");
  generate(gpt, [stoi["<start>"]!, stoi["the"]!], stoi["."]!, itos, vocabSize,
      blockSize, dummyEnc);
}
