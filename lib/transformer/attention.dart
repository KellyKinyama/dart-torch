import 'dart:math';
import '../nn/value_vector.dart';

ValueVector scaledDotProductAttention(
    ValueVector q, ValueVector k, ValueVector v) {
  final dk = k.values.length.toDouble();
  final score = q.dot(k) * (1.0 / sqrt(dk));
  // Since this is simplified for one query-key pair,
  // we just scale v by the score (ideal for single attention pair)
  return v * score;
}
