import 'dart:math';
import '../nn/value.dart';
import '../nn/value_vector.dart';
import 'transformer_block.dart';

void main() {
  final dModel = 8;
  final dFF = 16;
  final block = TransformerBlock(dModel, dFF);

  final input = ValueVector(List.generate(
    dModel,
    (_) => Value(Random().nextDouble()),
  ));

  final output = block.forward(input);
  print('Transformer Block Output:');
  print(output);
}
