import 'dart:math' as math;
import 'tensor.dart';
import 'aft_gpt.dart';

extension GPTGenerator on AFT_GPT {
  /// Generates a sequence of [maxNewTokens] given a starting prompt [idx]
  List<int> generate(List<int> idx, int maxNewTokens) {
    for (int i = 0; i < maxNewTokens; i++) {
      // 1. Crop current indices to the max blockSize supported by the model
      final inputIdx =
          idx.length > blockSize ? idx.sublist(idx.length - blockSize) : idx;

      // 2. Forward pass to get logits for the whole sequence
      final logits = forward(inputIdx);

      // 3. Focus only on the last time step [T-1, VocabSize]
      // We extract the scores for the very last token
      final int lastTokenOffset = (inputIdx.length - 1) * logits.shape[1];

      int nextToken = 0;
      double maxScore = double.negativeInfinity;

      // 4. Greedy Search: Pick the token with the highest logit score
      for (int v = 0; v < logits.shape[1]; v++) {
        double score = logits.data[lastTokenOffset + v];
        if (score > maxScore) {
          maxScore = score;
          nextToken = v;
        }
      }

      // 5. Append predicted token to the sequence
      idx.add(nextToken);

      // Optional: Stop if an <End Of String> token is generated
      // if (nextToken == eosTokenId) break;
    }
    return idx;
  }
}

void main() {
  final gpt = AFT_GPT(
      vocabSize: 100, embedSize: 64, blockSize: 32, numLayers: 4, numHeads: 4);

  // Start with a "prompt" (e.g., token IDs for "The cat")
  List<int> prompt = [1, 2];

  print('Prompt: $prompt');

  // Generate 10 new tokens
  final result = gpt.generate(prompt, 10);

  print('Generated Sequence: $result');
}
