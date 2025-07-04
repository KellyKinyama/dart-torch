// file: example_gpt_training.dart


// Import your core Value and Module system
import '/nn/value.dart';
import '/nn/value_vector.dart';
// Import your TransformerDecoder
import 'transformer_decoder.dart'; // Your existing TransformerDecoder
// Ensure all sub-modules of TransformerDecoder are accessible:
// transformer_decoder_block.dart, multi_head_attention.dart,
// feed_forward.dart, layer_norm2.dart, self_attention.dart,
// cross_attention.dart, multi_head_cross_attention.dart

// Re-using a simple SGD optimizer
class SGD {
  final List<Value> parameters;
  final double learningRate;

  SGD(this.parameters, this.learningRate);

  void step() {
    for (final p in parameters) {
      // Only update if gradient exists
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
  print("--- Generative Pretrained Transformer (GPT) Training Example ---");

  // 1. Define GPT Model Hyperparameters
  const int vocabSize = 40; // Increased vocabulary size
  const int embedSize = 32;
  const int blockSize = 15; // Increased block size for longer sequences
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  // 2. Expanded Vocabulary for demonstration
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
    "dart": 19,
    "programming": 20,
    "language": 21,
    "example": 22,
    "code": 23,
    "learning": 24,
    "machine": 25,
    "deep": 26,
    "neural": 27,
    "networks": 28,
    "great": 29,
    "simple": 30,
    "powerful": 31,
    "today": 32,
    "future": 33,
    "data": 34,
    "science": 35,
    "artificial": 36,
    "intelligence": 37,
    "next": 38,
    "token": 39,
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // Verify vocabSize covers all tokens
  assert(stoi.length <= vocabSize,
      "vocabSize is too small for the defined vocabulary.");

  // Get special token IDs
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;
  final int endTokenId = stoi["."]!; // Using '.' as an example end token

  print("\nExample Vocabulary:");
  print(itos);

  // 3. Create a Dummy Dataset with more varied sequences
  final List<List<int>> rawSequences = [
    [startTokenId, stoi["hello"]!, stoi["world"]!, stoi["."]!],
    [
      startTokenId,
      stoi["this"]!,
      stoi["is"]!,
      stoi["a"]!,
      stoi["test"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["the"]!,
      stoi["quick"]!,
      stoi["brown"]!,
      stoi["fox"]!,
      stoi["jumps"]!,
      stoi["over"]!,
      stoi["the"]!,
      stoi["lazy"]!,
      stoi["dog"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["dart"]!,
      stoi["is"]!,
      stoi["a"]!,
      stoi["great"]!,
      stoi["programming"]!,
      stoi["language"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["learning"]!,
      stoi["deep"]!,
      stoi["neural"]!,
      stoi["networks"]!,
      stoi["is"]!,
      stoi["powerful"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["machine"]!,
      stoi["learning"]!,
      stoi["example"]!,
      stoi["code"]!,
      stoi["."]!
    ],
    [
      startTokenId,
      stoi["artificial"]!,
      stoi["intelligence"]!,
      stoi["is"]!,
      stoi["the"]!,
      stoi["future"]!,
      stoi["."]!
    ],
  ];

  List<List<int>> trainInputs = [];
  List<List<int>> trainTargets = [];

  for (var seq in rawSequences) {
    // Input sequence: all tokens except the last one
    List<int> input = seq.sublist(0, seq.length - 1);
    // Target sequence: all tokens except the first one (what we predict)
    List<int> target = seq.sublist(1);

    // Pad or truncate sequences to blockSize
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

  print("\nDummy Training Data:");
  for (int i = 0; i < trainInputs.length; i++) {
    print("  Input:  ${trainInputs[i].map((id) => itos[id]).join(' ')}");
    print("  Target: ${trainTargets[i].map((id) => itos[id]).join(' ')}");
  }

  // 4. Instantiate the GPT model (your TransformerDecoder)
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

  // 5. Setup Optimizer
  const double learningRate = 0.01;
  final optimizer = SGD(gptModel.parameters(), learningRate);
  print("Optimizer (SGD) initialized with learning rate: $learningRate");

  final List<ValueVector> dummyEncoderOutput = List.generate(
    1,
    (_) => ValueVector(List.filled(embedSize, Value(0.0))),
  );

  // 6. Training Loop
  const int numEpochs = 1000; // Increased epochs for more complex data
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

    if ((epoch + 1) % 100 == 0 || epoch == 0) {
      // Print less frequently for more epochs
      print(
          "Epoch ${epoch + 1}/${numEpochs}, Loss: ${totalLoss / trainInputs.length}");
    }
  }

  print("\n--- Training Complete ---");

  // 7. Test Generation after (pseudo) training
  print("\n--- Testing Generation After Training ---");
  List<int> generatedSequence = [startTokenId];
  final int maxTestGenerationLength = 20; // Allow longer generation

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

  print("Generated Text: ${generatedSequence.map((id) => itos[id]).join(' ')}");
  print("---------------------------------------");
}
