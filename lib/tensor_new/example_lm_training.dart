import '../tensor/tensor.dart';
import 'module.dart';
import 'transformer_encoder.dart';
import 'layer.dart'; // Assuming your Layer/Module classes are here

// 1. Define the LMHead Module
class LMHead extends Module {
  final Layer projection;

  LMHead(int embedSize, int vocabSize)
      : projection = Layer(embedSize, vocabSize, useGelu: false);

  Tensor forward(Tensor x) => projection.forward(x);

  @override
  List<Tensor> parameters() => projection.parameters();

  void zeroGrad() {
    for (var p in parameters()) {
      p.grad.fillRange(0, p.grad.length, 0.0);
    }
  }
}

void main() {
  print("--- Transformer LM Head Training Example ---");

  // Hyperparameters
  const vocabSize = 20;
  const embedSize = 32;
  const blockSize = 10;
  const learningRate = 0.01;

  // 2. Initialize Encoder and Head
  final encoder = TransformerEncoder(
    vocabSize: vocabSize,
    embedSize: embedSize,
    blockSize: blockSize,
    numLayers: 2,
    numHeads: 4,
  );

  final lmHead = LMHead(embedSize, vocabSize);

  // 3. Sample Input: [1, 5, 8, 2, 1, 9]
  final sampleInputSequence = [1, 5, 8, 2, 1, 9];

  // 4. Initial Forward Pass
  final embeddings = encoder.forward(sampleInputSequence);
  final logits = lmHead.forward(embeddings); // Shape: [6, 20]

  // Slice out only the first token's prediction logits [1, 20]
  final firstTokenLogits = logits.slice2D(1, vocabSize);

  // 5. Training Step
  print("\n--- Training Step ---");

  // Dummy target: In a real scenario, this might be a one-hot vector for the next word
  final targetLogits = Tensor.random([1, vocabSize]);

  final initialLoss = firstTokenLogits.mseLoss(targetLogits);
  print("Initial Loss: ${initialLoss.data[0].toStringAsFixed(6)}");

  // 6. Backprop
  encoder.zeroGrad();
  lmHead.zeroGrad();
  initialLoss.backward();

  // 7. Optimizer Step (Update all parameters)
  final allParams = [...encoder.parameters(), ...lmHead.parameters()];
  for (var p in allParams) {
    for (int i = 0; i < p.data.length; i++) {
      p.data[i] -= learningRate * p.grad[i];
    }
  }

  // 8. Verification Forward Pass
  final embeddingsAfter = encoder.forward(sampleInputSequence);
  final logitsAfter = lmHead.forward(embeddingsAfter);
  final firstTokenLogitsAfter = logitsAfter.slice2D(1, vocabSize);

  final lossAfter = firstTokenLogitsAfter.mseLoss(targetLogits);
  print("Loss After Update: ${lossAfter.data[0].toStringAsFixed(6)}");

  if (lossAfter.data[0] < initialLoss.data[0]) {
    print("SUCCESS: The model is learning to predict the target logits!");
  }
}
