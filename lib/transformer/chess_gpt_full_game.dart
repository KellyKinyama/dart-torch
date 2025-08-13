// file: chess_gpt_full_game.dart

import 'dart:math';
import 'package:bishop/bishop.dart';
import '/nn/value.dart';
import '/nn/value_vector.dart';
import 'transformer_decoder.dart';

const NUM_SQUARES = 64;
// Simplified move representation: (from_square_idx * NUM_SQUARES) + to_square_idx
// This results in 64*64 = 4096 possible (start, end) pairs, many of which are illegal.
// int policyToModel(int from_square_idx, int to_square_idx) {
//   return (from_square_idx * NUM_SQUARES) + to_square_idx;
// }

// (int, int) policyToModel(int move_idx) {
//   // int policyToModel=(from_square_idx * NUM_SQUARES) + to_square_idx;
//   //(policyToModel-to_square_idx)/NUM_SQUARES=from_square_idx ;
//   int from_square_idx = (move_idx - to_square_idx) / NUM_SQUARES;
//   return (from_square_idx * NUM_SQUARES) + to_square_idx;
// }

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, learningRate) : this.learningRate = learningRate;

  void step() {
    for (final p in parameters) {
      if (p.grad != null) {
        p.data -= learningRate * p.grad!;
      }
    }
  }

  void zeroGrad() {
    for (final p in parameters) {
      p.grad = 0;
    }
  }
}

// Global helper function to generate a vocabulary from a dataset of games.
Map<String, int> _generateChessVocab(List<List<String>> gameSequences) {
  final Map<String, int> stoi = {};
  int idCounter = 0;

  stoi['<start>'] = idCounter++;
  stoi['<end>'] = idCounter++;
  stoi['<pad>'] = idCounter++;

  for (final game in gameSequences) {
    for (final move in game) {
      if (!stoi.containsKey(move)) {
        stoi[move] = idCounter++;
      }
    }
  }

  return stoi;
}

void main() async {
  print("--- Generative Pretrained Transformer (GPT) for Full Chess Games ---");

  // A small dataset of complete games for demonstration
  final List<List<String>> rawGameSequences = [
    [
      'e2e4',
      'e7e5',
      'g1f3',
      'b8c6',
      'f1c4',
      'f8c5',
      'c2c3',
      'g8f6',
      'd2d4',
      'e5d4',
      'c3d4',
      'c4b4'
    ],
    [
      'd2d4',
      'd7d5',
      'c2c4',
      'e7e6',
      'c1f4',
      'g8f6',
      'g1f3',
      'c7c5',
      'c4d5',
      'f6d5',
      'e3e4',
      'd5f4',
      'd1d8'
    ],
  ];

  final Map<String, int> stoi = _generateChessVocab(rawGameSequences);
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));
  final int vocabSize = stoi.length;
  // final int vocabSize = 4096;
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;
  final int endTokenId = stoi["<end>"]!;

  const int embedSize = 32;
  const int blockSize = 16;
  // const int blockSize = 64;
  const int numLayers = 4;
  const int numHeads = 8;

  final gptModel = TransformerDecoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: numLayers,
    numHeads: numHeads,
    encoderEmbedSize: embedSize,
  );
  final optimizer = SGD(gptModel.parameters(), 0.01);
  final List<ValueVector> dummyEncoderOutput = List.generate(
    1,
    (_) => ValueVector(List.filled(embedSize, Value(0.0))),
  );

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

  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 4. Setup Optimizer
  const double learningRate = 0.01;

  // ... (Training logic is omitted for brevity as it's the same as before) ...
  // For this example, let's assume the model is already trained.
  train(
      trainInputs, trainTargets, optimizer, gptModel, dummyEncoderOutput, stoi);
  // --- Playing a Full Game ---
  print("\n--- Starting a Full Game ---");
  final Game chessGame = Game();
  List<int> gameHistoryTokens = [startTokenId];

  while (!chessGame.gameOver) {
    if (gameHistoryTokens.last == endTokenId) break;

    // Use a limited history as input to the model (up to blockSize)
    List<int> currentInput = List.from(gameHistoryTokens);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    final List<ValueVector> logits =
        gptModel.forward(currentInput, dummyEncoderOutput);
    final ValueVector lastTokenLogits = logits.last;

    // Get legal moves from the current game state
    final List<Move> legalMoves = chessGame.generateLegalMoves();
    final Set<int> legalMoveTokenIds = legalMoves
        .map((m) => stoi[chessGame.toAlgebraic(m)])
        .whereType<int>()
        .toSet();

    // Mask illegal moves
    final List<Value> maskedLogits = List.generate(vocabSize, (index) {
      if (legalMoveTokenIds.contains(index)) {
        return lastTokenLogits.values[index];
      }
      return Value(-double.infinity); // Set illegal move probabilities to zero
    });

    // maskedLogits.removeWhere((element) => element.data == -double.infinity);

    // print("Gpt model probabilities: $maskedLogits");

    // Apply Softmax to get probabilities
    final ValueVector probabilities = ValueVector(maskedLogits).softmax();

    // final numbers = <String>['one', 'two', 'three', 'four'];
    // numbers.removeWhere((item) => item.length == 3);
    // print(numbers); // [three, four]

    // probabilities.values.removeWhere((element) => element.data == 0.0);

    // Greedily select the move with the highest probability
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    // Check for a valid move
    if (predictedNextToken == -1) {
      print(
          "Model failed to suggest a legal move. Game over. ${predictedNextToken}");
      break;
    }

    // Update game state
    final String nextMoveAlgebraic = itos[predictedNextToken]!;
    final Move? move = chessGame.getMove(nextMoveAlgebraic);
    // print("Move: $move");

    if (move != null) {
      print("Move: ${chessGame.toAlgebraic(move)}");
      chessGame.makeMove(move);
      gameHistoryTokens.add(predictedNextToken);
    } else {
      if (predictedNextToken == startTokenId) {
        gameHistoryTokens.add(predictedNextToken);
      } else {
        print(
            "Model suggested an illegal move: $nextMoveAlgebraic. Game over.");

        break;
      }
    }
  }

  print("\n--- Game Over ---");
  print(chessGame.result?.readable ?? 'Unknown result');
}

void train(
    List<List<int>> trainInputs,
    List<List<int>> trainTargets,
    SGD optimizer,
    TransformerDecoder gptModel,
    final List<ValueVector> dummyEncoderOutput,
    Map<String, int> stoi) {
  final int padTokenId = stoi["<pad>"]!;
  // 5. Training Loop
  const int numEpochs = 200;
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

    if ((epoch + 1) % 1 == 0 || epoch == 0) {
      print(
          "Epoch ${epoch + 1}/${numEpochs}, Loss: ${totalLoss / trainInputs.length}");
    }
  }
  print("\n--- Training Complete ---");
}
