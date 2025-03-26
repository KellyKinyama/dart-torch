import 'dart:math';
import '../nn/value.dart';
import '../nn/value_vector.dart';

class PositionalEmbedding {
  final int maxLen;
  final int dim;
  final List<ValueVector> embeddings;

  PositionalEmbedding(this.maxLen, this.dim)
      : embeddings = List.generate(
          maxLen,
          (pos) => ValueVector(List.generate(dim, (i) {
            final angle = pos / pow(10000, 2 * (i ~/ 2) / dim);
            return Value(i.isEven ? sin(angle) : cos(angle));
          })),
        );

  ValueVector get(int pos) => embeddings[pos];
}
