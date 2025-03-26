import '../nn/value.dart';
import 'encoder.dart';
import 'classifier_head.dart';
import 'positional_embedding.dart';
import '../nn/value_vector.dart';
import 'cross_entropy_loss.dart';

void trainModel(List<List<int>> inputTokens, List<int> labels,
    int vocabSize, int maxLen, int dim, int heads, int ffDim, int epochs) {
  final embedding = List.generate(vocabSize,
      (_) => ValueVector(List.generate(dim, (_) => Value.random())));
  final posEmbedding = PositionalEmbedding(maxLen, dim);
  final encoder = TransformerEncoder(dim, heads, ffDim);
  final head = ClassifierHead(dim, vocabSize);

  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalLoss = 0.0;
    for (int i = 0; i < inputTokens.length; i++) {
      final tokens = inputTokens[i];
      final label = labels[i];

      var x = ValueVector(List.generate(dim, (_) => Value(0.0)));
      for (int t = 0; t < tokens.length; t++) {
        final tokenVec = embedding[tokens[t]] + posEmbedding.get(t);
        x = encoder.forward(tokenVec, [tokenVec]); // simple self-attention
      }

      final logits = head.forward(x);
      final loss = crossEntropyLoss(logits, label);
      totalLoss += loss.data;

      // Backprop (loss.backward already called in crossEntropy)
      loss.backward();

      // Simple SGD step
      for (final param in [...embedding.expand((e) => e.values), ...encoder.mha.getParameters(), ...encoder.ff1.getParameters(), ...encoder.ff2.getParameters(), ...head.fc.getParameters()]) {
        param.data -= 0.01 * param.grad;
        param.grad = 0.0;
      }
    }

    print('Epoch $epoch, Loss: ${totalLoss / inputTokens.length}');
  }
}
