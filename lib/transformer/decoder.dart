import 'dart:convert';
import 'dart:io';
import 'dart:math';
import '../nn/value_vector.dart';
import 'layer_norm.dart';
// import 'multi_head_attention.dart';
import 'transformer_block.dart';
// import 'encoder.dart';

class TransformerDecoder {
  final List<TransformerBlock> layers;
  final LayerNorm finalNorm;

  TransformerDecoder(int numLayers, int embedDim, int numHeads, int ffDim)
      : layers = List.generate(
            numLayers,
            (_) => TransformerBlock(embedDim, numHeads, ffDim)),
        finalNorm = LayerNorm(embedDim);

  List<ValueVector> call(List<ValueVector> x, List<ValueVector> encoderOutput) {
    for (var layer in layers) {
      x = layer.forward(x);
    }
    return x.map((v) => finalNorm.forward(v)).toList();
  }
}

List<double> softmax(List<double> logits) {
  final maxLogit = logits.reduce(max);
  final exps = logits.map((l) => exp(l - maxLogit)).toList();
  final sumExp = exps.reduce((a, b) => a + b);
  return exps.map((e) => e / sumExp).toList();
}

void printPredictions(List<double> probs, List<String> labels) {
  final indexed =
      List.generate(probs.length, (i) => MapEntry(labels[i], probs[i]));
  indexed.sort((a, b) => b.value.compareTo(a.value));
  for (var entry in indexed) {
    print('${entry.key}: ${(entry.value * 100).toStringAsFixed(2)}%');
  }
}

void saveModelParams(String path, Map<String, dynamic> params) async {
  final jsonStr = jsonEncode(params);
  final file = File(path);
  await file.writeAsString(jsonStr);
  print('Model parameters saved to $path');
}

Future<Map<String, dynamic>> loadModelParams(String path) async {
  final file = File(path);
  if (!await file.exists()) throw Exception('Model file not found at $path');
  final content = await file.readAsString();
  return jsonDecode(content);
}
