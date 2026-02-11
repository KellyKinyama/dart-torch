import 'tensor.dart';
import 'aft_gpt.dart';
import 'optimizer.dart';

void main() {
  // 1. Setup Model Configuration
  final gpt = AFT_GPT(
      vocabSize: 100, // Size of your dictionary
      embedSize: 64, // Hidden dimension
      blockSize: 32, // Maximum sequence length
      numLayers: 4, // Number of Transformer blocks
      numHeads: 4 // Heads for attention
      );

  final optimizer = SGD(gpt.parameters(), lr: 0.01);

  // 2. Mock Input: A sequence of token IDs (e.g., "The cat sat")
  // Shape: List of integers (indices)
  final inputTokens = [1, 5, 12, 8, 3];

  // 3. Mock Target: The next tokens in the sequence
  // For training, we want the output to match some target distribution
  final target = Tensor.random([inputTokens.length, 100]); // [T, VocabSize]

  print('--- AFT-GPT Training Iteration ---');

  // 4. Forward Pass
  // This runs: Embedding -> Positional Encoding -> 4x AFT Blocks -> LayerNorm -> LM Head
  final logits = gpt.forward(inputTokens);
  print('Logits shape: ${logits.shape}'); // [5, 100]

  // 5. Loss Calculation
  final loss = logits.mseLoss(target);
  print('Step 0 Loss: ${loss.data[0].toStringAsFixed(6)}');

  // 6. Backward Pass & Optimization
  optimizer.zeroGrad();
  loss.backward();
  optimizer.step();

  // 7. Verification
  final nextLoss = gpt.forward(inputTokens).mseLoss(target);
  print('Step 1 Loss: ${nextLoss.data[0].toStringAsFixed(6)}');
}
