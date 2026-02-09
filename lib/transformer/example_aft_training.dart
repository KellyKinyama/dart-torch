// file: example_aft_training.dart

import '/nn/value.dart';
import '/nn/value_vector.dart';
// Import your new AFT-based TransformerDecoder
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
  print("--- Attention Free Transformer (AFT-GPT) Training Example ---");

  // 1. Define AFT-GPT Model Hyperparameters
  const int vocabSize = 20;
  const int embedSize = 32;
  const int blockSize = 10; // This is used as maxSeqLen for AFT biases
  const int numLayers = 3;
  const int numHeads = 4;

  // 2. Vocabulary setup (unchanged)
  final Map<String, int> stoi = {
    "hello": 0,
    "world": 1,
    "this": 2,
    "is": 3,
    "a": 4,
    "test": 5,
    "generation": 6,
    "model": 7,
    "the": 8,
    "quick": 9,
    "brown": 10,
    "fox": 11,
    "jumps": 12,
    "over": 13,
    "lazy": 14,
    "dog": 15,
    ".": 16,
    "<start>": 17,
    "<pad>": 18,
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;

  // 3. Create Dummy Dataset (unchanged)
  final List<List<int>> rawSequences = [
    [startTokenId, stoi["hello"]!, stoi["world"]!, stoi["."]!],
    [
      startTokenId,
      stoi["the"]!,
      stoi["quick"]!,
      stoi["brown"]!,
      stoi["fox"]!,
      stoi["."]!
    ],
  ];

  List<List<int>> trainInputs = [];
  List<List<int>> trainTargets = [];

  for (var seq in rawSequences) {
    List<int> input = seq.sublist(0, seq.length - 1);
    List<int> target = seq.sublist(1);
    while (input.length < blockSize) {
      input.add(padTokenId);
      target.add(padTokenId);
    }
    trainInputs.add(input);
    trainTargets.add(target);
  }

  // 4. Instantiate the AFT-GPT model
  print("\nInitializing AFT-GPT (TransformerDecoder)...");
  // Ensure your TransformerDecoder constructor now takes maxSeqLen
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize, // This initializes AFT position biases
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );

  print("Model initialized. Parameters: ${gptModel.parameters().length}");

  // 5. Setup Optimizer
  // AFT-full can sometimes handle higher learning rates due to
  // the sigmoid gating on the query providing inherent normalization.
  const double learningRate = 0.05;
  final optimizer = SGD(gptModel.parameters(), learningRate);

  // 6. Dummy Encoder Output (for Cross-Attention compatibility)
  final List<ValueVector> dummyEncoderOutput = List.generate(
    1,
    (_) => ValueVector(List.filled(embedSize, Value(0.0))),
  );

  // 7. Training Loop
  const int numEpochs = 200;
  for (int epoch = 0; epoch < numEpochs; epoch++) {
    double totalLoss = 0.0;

    for (int i = 0; i < trainInputs.length; i++) {
      optimizer.zeroGrad();

      // Forward Pass (Now using AFT logic internally)
      final List<ValueVector> logits =
          gptModel.forward(trainInputs[i], dummyEncoderOutput);

      // Compute Cross-Entropy Loss
      Value batchLoss = Value(0.0);
      int activeTokens = 0;

      for (int t = 0; t < logits.length; t++) {
        if (trainTargets[i][t] != padTokenId) {
          final int trueTargetId = trainTargets[i][t];
          final Value trueLogit = logits[t].values[trueTargetId];
          final Value logSumExp = logits[t]
              .values
              .map((v) => v.exp())
              .reduce((a, b) => a + b)
              .log();

          batchLoss += (logSumExp - trueLogit);
          activeTokens++;
        }
      }

      if (activeTokens > 0) {
        batchLoss = batchLoss / Value(activeTokens.toDouble());
        totalLoss += batchLoss.data;
        batchLoss.backward();
        optimizer.step();
      }
    }
    if ((epoch + 1) % 10 == 0) {
      print("Epoch ${epoch + 1}, Avg Loss: ${totalLoss / trainInputs.length}");
    }
  }

  // 8. Test Generation
  print("\nGenerated Text (AFT):");
  List<int> generated = [startTokenId];
  for (int i = 0; i < 10; i++) {
    final List<ValueVector> logits =
        gptModel.forward(generated, dummyEncoderOutput);
    final ValueVector probs = logits.last.softmax();

    // Greedy sample
    int nextToken = 0;
    double maxP = -1.0;
    for (int j = 0; j < probs.values.length; j++) {
      if (probs.values[j].data > maxP) {
        maxP = probs.values[j].data;
        nextToken = j;
      }
    }
    generated.add(nextToken);
    if (nextToken == stoi["."]) break;
  }
  print(generated.map((id) => itos[id]).join(' '));
}
