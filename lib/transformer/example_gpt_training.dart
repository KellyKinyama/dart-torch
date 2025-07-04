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
  const int vocabSize = 20; // Example vocabulary size
  const int embedSize = 32;
  const int blockSize = 10; // Maximum sequence length the GPT can process
  const int numLayers = 3;
  const int numHeads = 4;

  print("GPT Model Configuration:");
  print("  Vocabulary Size: $vocabSize");
  print("  Embedding Size: $embedSize");
  print("  Block Size (Max Context Length): $blockSize");
  print("  Number of Layers: $numLayers");
  print("  Number of Heads: $numHeads");

  // 2. Simple Vocabulary for demonstration
  // This vocabulary must be consistent between training and inference
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
    ".": 16, // End of sentence token
    "<start>": 17, // Start of sequence token
    "<pad>": 18, // Padding token
  };
  final Map<int, String> itos = stoi.map((key, value) => MapEntry(value, key));

  // Get special token IDs
  final int startTokenId = stoi["<start>"]!;
  final int padTokenId = stoi["<pad>"]!;

  print("\nExample Vocabulary:");
  print(itos);

  // 3. Create a Dummy Dataset
  // In a real scenario, this would be loaded from files, tokenized, and batched.
  // We'll create a few simple sequences for next-token prediction.
  // Each sequence is (input_tokens, target_tokens) where target_tokens are input_tokens shifted by one.
  // E.g., "hello world ." -> input: "<start> hello world", target: "hello world ."

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

    // Pad sequences to blockSize if needed (for simplicity, we'll keep them shorter or truncate)
    if (input.length > blockSize) {
      input = input.sublist(0, blockSize);
      target =
          target.sublist(0, blockSize); // Make sure target matches input length
    }
    // Pad if shorter than blockSize for consistent input shapes in a batch
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
    encoderEmbedSize:
        embedSize, // Still needed to satisfy constructor for cross-attention
  );
  print(
      "GPT (TransformerDecoder) initialized. Total parameters: ${gptModel.parameters().length}");

  // 5. Setup Optimizer
  const double learningRate = 0.01;
  final optimizer = SGD(gptModel.parameters(), learningRate);
  print("Optimizer (SGD) initialized with learning rate: $learningRate");

  // FIX: Provide a non-empty dummy encoder output to satisfy the CrossAttention layer.
  // In a true GPT, the CrossAttention layer would typically not exist or be ignored.
  // This dummy output allows the code to run without a "No element" error,
  // even though its values are not functionally meaningful for a pure GPT.
  final List<ValueVector> dummyEncoderOutput = List.generate(
    1, // Provide at least one dummy token
    (_) => ValueVector(List.filled(
        embedSize,
        Value(
            0.0))), // Each token vector should be of encoderEmbedSize (which is embedSize here)
  );

  // 6. Training Loop
  const int numEpochs = 500;
  print("\n--- Starting Training ---");

  for (int epoch = 0; epoch < numEpochs; epoch++) {
    double totalLoss = 0.0;

    for (int i = 0; i < trainInputs.length; i++) {
      final inputSequence = trainInputs[i];
      final targetSequence = trainTargets[i];

      // Zero gradients
      optimizer.zeroGrad();

      // Forward pass
      final List<ValueVector> logits =
          gptModel.forward(inputSequence, dummyEncoderOutput);

      // Calculate loss (Cross-Entropy Loss)
      // We are predicting the next token for each position in the input sequence.
      Value batchLoss = Value(0.0);
      int activeTokens = 0; // Count tokens that are not padding

      for (int t = 0; t < logits.length; t++) {
        // Only calculate loss for non-padding tokens
        if (targetSequence[t] != padTokenId) {
          final ValueVector tokenLogits = logits[t];
          final int trueTargetId = targetSequence[t];

          // Softmax then negative log likelihood for true target
          // This is a simplified cross-entropy calculation
          final Value trueLogit = tokenLogits.values[trueTargetId];
          final Value sumExpLogits =
              tokenLogits.values.map((v) => v.exp()).reduce((a, b) => a + b);
          final Value logSumExp = sumExpLogits.log();
          final Value negLogProb =
              logSumExp - trueLogit; // Negative log-likelihood

          batchLoss += negLogProb;
          activeTokens++;
        }
      }

      // Average loss over active tokens
      if (activeTokens > 0) {
        batchLoss = batchLoss / Value(activeTokens.toDouble());
      } else {
        batchLoss = Value(0.0); // No active tokens, no loss
      }

      totalLoss += batchLoss.data;

      // Backward pass
      batchLoss.backward();

      // Update parameters
      optimizer.step();
    }

    if ((epoch + 1) % 1 == 0 || epoch == 0) {
      print(
          "Epoch ${epoch + 1}/${numEpochs}, Loss: ${totalLoss / trainInputs.length}");
    }
  }

  print("\n--- Training Complete ---");

  // 7. Test Generation after (pseudo) training
  print("\n--- Testing Generation After Training ---");
  List<int> generatedSequence = [startTokenId];
  final int maxTestGenerationLength = 10;

  for (int i = 0; i < maxTestGenerationLength; i++) {
    List<int> currentInput = List.from(generatedSequence);
    if (currentInput.length > blockSize) {
      currentInput = currentInput.sublist(currentInput.length - blockSize);
    }

    // Pass dummy encoder output as before
    final List<ValueVector> logits =
        gptModel.forward(currentInput, dummyEncoderOutput);

    // Get the logits for the last token and sample
    final ValueVector lastTokenLogits = logits.last;
    final ValueVector probabilities = lastTokenLogits.softmax();

    // Greedy sampling for simplicity
    double maxProb = -1.0;
    int predictedNextToken = -1;
    for (int j = 0; j < probabilities.values.length; j++) {
      if (probabilities.values[j].data > maxProb) {
        maxProb = probabilities.values[j].data;
        predictedNextToken = j;
      }
    }

    generatedSequence.add(predictedNextToken);

    if (predictedNextToken == stoi["."]) {
      // Stop on sentence end token
      break;
    }
    if (generatedSequence.length >= maxTestGenerationLength + 1) {
      // +1 for start token
      break;
    }
  }

  print("Generated Text: ${generatedSequence.map((id) => itos[id]).join(' ')}");
  print("---------------------------------------");
}
