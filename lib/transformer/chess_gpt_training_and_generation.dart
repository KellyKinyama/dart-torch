// file: chess_gpt_training_and_generation.dart

import 'dart:math';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';
import 'package:bishop/bishop.dart'; // Using the Bishop library for chess moves

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      if (p.grad != null) {
        p.data -= learningRate * p.grad!;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0; // Use null for no gradient
    }
  }
}

// Corrected function to generate a chess vocabulary from the training data.
Map<String, int> _generateChessVocab(List<List<String>> gameSequences) {
  final Map<String, int> stoi = {};
  int idCounter = 0;

  // Add special tokens
  stoi['<start>'] = idCounter++;
  stoi['<end>'] = idCounter++;
  stoi['<pad>'] = idCounter++;

  // Populate vocabulary with all unique moves from the training data
  for (final game in gameSequences) {
    for (final move in game) {
      if (!stoi.containsKey(move)) {
        stoi[move] = idCounter++;
      }
    }
  }

  return stoi;
}

void main() {
  print("--- Generative Pretrained Transformer (GPT) for Chess Moves ---");

  // 1. Create a Dummy Dataset of Chess Games (as move sequences) first
  // These are example game sequences. A real model would need thousands of games.
  final List<List<String>> rawGameSequences = [
    ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'c2c3', 'g8f6'],
    ['d2d4', 'd7d5', 'c2c4', 'c7c6', 'c1f4', 'b8d7', 'e2e3', 'g8f6'],
    ['g1f3', 'g8f6', 'c2c4', 'e7e6', 'd2d4', 'b7b6', 'e2e3', 'c8b7'],
  ];

  // 2. Define Model Hyperparameters
  // Now, we can define the vocabulary from the full training data
  final Map<String, int> stoi = _generateChessVocab(rawGameSequences);
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));
  final int vocabSize = stoi.length;
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;
  final int endTokenId = stoi["<end>"]!;

  const int embedSize = 32;
  const int blockSize = 15; // Max sequence length (game history)
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  print("\nExample Chess Move Vocabulary (first 10 moves):");
  itos.keys.take(10).forEach((id) => print('  $id: ${itos[id]}'));

  List<List<int>> trainInputs = [];
  List<List<int>> trainTargets = [];

  for (var gameSeq in rawGameSequences) {
    // Convert algebraic moves to integer IDs
    final List<int> tokenizedSeq = [
      startTokenId,
      ...gameSeq.map((move) => stoi[move]!),
      endTokenId
    ];

    List<int> input = tokenizedSeq.sublist(0, tokenizedSeq.length - 1);
    List<int> target = tokenizedSeq.sublist(1);

    // Pad sequences to the block size
    if (input.length > blockSize) {
      input = input.sublist(0, blockSize);
      target = target.sublist(0, blockSize);
    }
    while (input.length < blockSize) {
      input.add(padTokenId);
      target.add(padTokenId);
    }
    trainInputs.add(input);
    trainTargets.add(target);
  }

  print("\nDummy Training Data (first game):");
  print("  Input:  ${trainInputs.first.map((id) => itos[id]).join(' ')}");
  print("  Target: ${trainTargets.first.map((id) => itos[id]).join(' ')}");

  // 3. Instantiate the GPT model (your TransformerDecoder)
  print("\nInitializing GPT (TransformerDecoder) for training...");
  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );
  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 4. Setup Optimizer
  const double learningRate = 0.01;
  final optimizer = SGD(gptModel.parameters(), learningRate);
  print("Optimizer (SGD) initialized with learning rate: $learningRate");

  final List<ValueVector> dummyEncoderOutput = List.generate(
    1,
    (_) => ValueVector(List.filled(embedSize, Value(0.0))),
  );

  // 5. Training Loop
  const int numEpochs = 250;
  print("\n--- Starting Training ---");
  for (int epoch = 0; epoch < numEpochs; epoch++) {
    double totalLoss = 0.0;
    for (int i = 0; i < trainInputs.length; i++) {
      final inputSequence = trainInputs[i];
      final targetSequence = trainTargets[i];
      optimizer.zeroGrad();
      final List<ValueVector> logits =
          gptModel.forward(inputSequence, dummyEncoderOutput);
      Value batchLoss = Value(0.0);
      int activeTokens = 0;
      for (int t = 0; t < logits.length; t++) {
        if (targetSequence[t] != padTokenId) {
          final ValueVector tokenLogits = logits[t];
          final int trueTargetId = targetSequence[t];
          final Value trueLogit = tokenLogits.values[trueTargetId];
          final Value sumExpLogits =
              tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
          final Value logSumExp = sumExpLogits.log();
          final Value negLogProb = logSumExp - trueLogit;
          batchLoss += negLogProb;
          activeTokens++;
        }
      }
      if (activeTokens > 0) {
        batchLoss = batchLoss / Value(activeTokens.toDouble());
      } else {
        batchLoss = Value(0.0);
      }
      totalLoss += batchLoss.data;
      batchLoss.backward();
      optimizer.step();
    }

    if ((epoch + 1) % 25 == 0 || epoch == 0) {
      print(
          "Epoch ${epoch + 1}/${numEpochs}, Loss: ${totalLoss / trainInputs.length}");
    }
  }
  print("\n--- Training Complete ---");

  // 6. Test Generation after (pseudo) training
  print("\n--- Testing Generation After Training ---");
  List<int> generatedSequence = [startTokenId];
  final int maxTestGenerationLength = 20;

  for (int i = 0; i < maxTestGenerationLength; i++) {
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    final List<ValueVector> logits =
        gptModel.forward(currentInput, dummyEncoderOutput);

    final ValueVector lastTokenLogits = logits.last;
    final ValueVector probabilities = lastTokenLogits.softmax();

    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    generatedSequence.add(predictedNextToken);

    if (predictedNextToken == endTokenId) {
      print("End of sequence token detected.");
      break;
    }
    if (generatedSequence.length >= maxTestGenerationLength + 1) {
      print("Maximum generation length reached.");
      break;
    }
  }

  print(
      "Generated Moves: ${generatedSequence.map((id) => itos[id]).join(' ')}");
  print("---------------------------------------");
}
