import 'dart:math' as math;
import 'tensor.dart';
import 'aft_gpt.dart';
import 'optimizer.dart';

class CharacterTokenizer {
  late Map<String, int> charToId;
  late Map<int, String> idToChar;

  CharacterTokenizer(String text) {
    final chars = text.split('').toSet().toList()..sort();
    charToId = {for (int i = 0; i < chars.length; i++) chars[i]: i};
    idToChar = {for (int i = 0; i < chars.length; i++) i: chars[i]};
  }

  List<int> encode(String text) =>
      text.split('').map((c) => charToId[c]!).toList();
  String decode(List<int> ids) => ids.map((i) => idToChar[i]!).join('');
  int get vocabSize => charToId.length;
}

void main() {
  final random = math.Random();

  // --- Stage 1: Data Preparation ---
  final data = "the cat sat on the mat. the dog sat on the log.";
  final tokenizer = CharacterTokenizer(data);
  final encodedData = tokenizer.encode(data);

  // --- Stage 2: Model Setup ---
  final blockSize = 8;
  final gpt = AFT_GPT(
    vocabSize: tokenizer.vocabSize,
    embedSize: 32,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 4,
  );

  // Use a slightly lower LR and ensure SGD includes the clipping we discussed
  final optimizer = SGD(gpt.parameters(), lr: 0.01);

  print('--- Training Phase ---');
  for (int epoch = 0; epoch <= 100; epoch++) {
    double totalLoss = 0;

    for (int i = 0; i < encodedData.length - blockSize - 1; i++) {
      final inputIdx = encodedData.sublist(i, i + blockSize);
      final targetIds = encodedData.sublist(i + 1, i + blockSize + 1);

      final targetTensor = Tensor.fill([blockSize, tokenizer.vocabSize], 0.0);
      for (int t = 0; t < blockSize; t++) {
        targetTensor.data[t * tokenizer.vocabSize + targetIds[t]] = 1.0;
      }

      optimizer.zeroGrad();
      final logits = gpt.forward(inputIdx);
      final loss = logits.mseLoss(targetTensor);
      loss.backward();
      optimizer.step();

      totalLoss += loss.data[0];
    }

    if (epoch % 20 == 0) {
      print(
          'Epoch $epoch | Avg Loss: ${(totalLoss / encodedData.length).toStringAsFixed(4)}');
    }
  }

  // --- Stage 3: Generation Phase ---
  print('\n--- Generation Phase (Sampling Enabled) ---');
  String prompt = "the cat ";
  List<int> generated = tokenizer.encode(prompt);
  double temperature = 0.8; // Lower = more predictable, Higher = more creative

  for (int i = 0; i < 20; i++) {
    final input = generated.length > blockSize
        ? generated.sublist(generated.length - blockSize)
        : generated;

    final logits = gpt.forward(input);

    // 1. Extract only the logits for the VERY LAST token [1, VocabSize]
    final int V = tokenizer.vocabSize;
    final int lastRowOffset = (input.length - 1) * V;
    final lastTokenLogits = Tensor([1, V]);

    for (int v = 0; v < V; v++) {
      // Apply temperature scaling before softmax
      lastTokenLogits.data[v] = logits.data[lastRowOffset + v] / temperature;
    }

    // 2. Use your built-in Softmax
    final probs = lastTokenLogits.softmax();

    // 3. Multinomial Sampling (Pick based on probability)
    int nextId = 0;
    double r = random.nextDouble();
    double cumulative = 0.0;

    for (int v = 0; v < V; v++) {
      cumulative += probs.data[v];
      if (r <= cumulative) {
        nextId = v;
        break;
      }
    }

    generated.add(nextId);
  }

  print('Prompt: "$prompt"');
  print('Result: "${tokenizer.decode(generated)}"');
}
