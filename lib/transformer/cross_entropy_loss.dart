import '../nn/value_vector.dart';
import '../nn/value.dart';

Value crossEntropyLoss(ValueVector logits, int targetIndex) {
  final maxLogit = logits.values.reduce((a, b) => a.data > b.data ? a : b);
  final expLogits = logits.values.map((v) => (v - maxLogit).exp()).toList();
  // final sumExp = expLogits.sum();
  Value sumExp = expLogits.reduce((a, b) => a + b);
  final probs = ValueVector(expLogits) / sumExp;

  return -probs.values[targetIndex].log();
}
