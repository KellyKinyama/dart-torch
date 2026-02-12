import 'dart:math' as math;
import 'dart:typed_data';
import '../tensor/tensor.dart';
import 'transformer_decoder.dart';

// --- 1. SGD Optimizer ---
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

// --- 2. Fused Cross Entropy Loss (Stable) ---
Tensor crossEntropy(Tensor logits, List<int> targets, int vocabSize) {
  int numTokens = targets.length;
  double totalLoss = 0;

  // Forward: Log-Sum-Exp trick for stability
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
      double logSumExp = maxL + math.log(sumExp + 1e-12);

      for (int v = 0; v < vocabSize; v++) {
        double prob = math.exp(logits.data[offset + v] - logSumExp);
        double targetSignal = (v == target) ? 1.0 : 0.0;
        logits.grad[offset + v] += (prob - targetSignal) * gradFromLoss;
      }
    }
  };
  return loss;
}

// --- 3. Main Training & Generation Loop ---
void main() {
  print("--- GPT Full Cycle: Training + Stochastic Generation ---");

  // Hyperparameters
  const int vocabSize = 25;
  const int embedSize = 32;
  const int blockSize = 16;
  final gpt = TransformerDecoder(
      vocabSize: vocabSize,
      embedSize: embedSize,
      blockSize: blockSize,
      numLayers: 2,
      numHeads: 4);

  final optimizer = SGD(gpt.parameters(), 0.2); // Strong LR for small data
  final dummyEnc = Tensor.zeros([1, embedSize]);

  // Vocabulary
  final Map<String, int> stoi = {
    "the": 0,
    "quick": 1,
    "brown": 2,
    "fox": 3,
    "jumps": 4,
    "over": 5,
    "lazy": 6,
    "dog": 7,
    ".": 8,
    "is": 9,
    "a": 10,
    "happy": 11,
    "<start>": 12,
  };
  final itos = stoi.map((k, v) => MapEntry(v, k));

  // Data: A single sequence to memorize
  String corpus = "the quick brown fox jumps over the lazy dog .";
  List<int> tokens = corpus.split(" ").map((t) => stoi[t]!).toList();

  // --- Training ---
  print("\nPhase 1: Overfitting the mini-corpus...");
  for (int epoch = 0; epoch <= 1000; epoch++) {
    optimizer.zeroGrad();

    // Shifted inputs/targets for Next Token Prediction
    List<int> inputs = tokens.sublist(0, tokens.length - 1);
    List<int> targets = tokens.sublist(1, tokens.length);

    final logits = gpt.forward(inputs, dummyEnc);
    final loss = crossEntropy(logits, targets, vocabSize);

    loss.backward();
    optimizer.step();

    if (epoch % 200 == 0) {
      print("Epoch $epoch | Loss: ${loss.data[0].toStringAsFixed(6)}");
    }
  }

  // --- Generation with Temperature ---
  print("\nPhase 2: Stochastic Generation");
  List<int> genSeq = [stoi["the"]!, stoi["quick"]!];
  double temperature = 0.8; // Lower = more confident, Higher = more random
  final rand = math.Random();

  print("Prompt: ${genSeq.map((id) => itos[id]).join(' ')}");

  for (int i = 0; i < 10; i++) {
    // Forward pass
    final logits = gpt.forward(genSeq, dummyEnc);

    // Get logits for the last token only
    int lastIdx = genSeq.length - 1;
    int offset = lastIdx * vocabSize;

    // Extract row and apply temperature
    List<double> rowProbs = [];
    double maxL = -double.infinity;
    for (int v = 0; v < vocabSize; v++)
      if (logits.data[offset + v] > maxL) maxL = logits.data[offset + v];

    double sumExp = 0;
    List<double> exps = [];
    for (int v = 0; v < vocabSize; v++) {
      double e = math.exp((logits.data[offset + v] - maxL) / temperature);
      exps.add(e);
      sumExp += e;
    }

    // Stochastic Sampling (Weighted Random)
    double r = rand.nextDouble();
    double cumulative = 0;
    int nextId = stoi["."]!; // Fallback

    for (int v = 0; v < vocabSize; v++) {
      cumulative += (exps[v] / sumExp);
      if (r <= cumulative) {
        nextId = v;
        break;
      }
    }

    genSeq.add(nextId);
    if (nextId == stoi["."]) break;

    // Keep context window within blockSize
    if (genSeq.length >= blockSize) break;
  }

  print("\nFinal Result:");
  print(genSeq.map((id) => itos[id] ?? "??").join(' '));
  print("------------------------------------------");
}
