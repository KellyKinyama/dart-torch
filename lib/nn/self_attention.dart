import 'dart:math';

import 'matrix2d.dart';
import 'value_vector.dart';

class LinearLayer {
  final Matrix2d weights;
  final ValueVector bias;

  LinearLayer(int inFeatures, int outFeatures)
      : weights = Matrix2d(outFeatures, inFeatures),
        bias = ValueVector(List.generate(outFeatures, (_) => Value(0)));

  ValueVector forward(ValueVector input) {
    var output = (weights * input.toMatrix()).toVector();
    return output + bias;
  }
}

class MultiHeadAttention {
  final int numHeads;
  final List<SelfAttention> heads;
  final LinearLayer outputLayer;

  MultiHeadAttention(int embedSize, this.numHeads)
      : heads = List.generate(numHeads, (_) => SelfAttention(embedSize ~/ numHeads)),
        outputLayer = LinearLayer(embedSize, embedSize);

  Matrix2d forward(Matrix2d input) {
    List<Matrix2d> headOutputs = heads.map((head) => head.forward(input)).toList();
    Matrix2d concatenated = concatenate(headOutputs);
    return outputLayer.forward(concatenated.toVector()).toMatrix();
  }
}

class SelfAttention {
  final LinearLayer queryLayer;
  final LinearLayer keyLayer;
  final LinearLayer valueLayer;

  SelfAttention(int embedSize)
      : queryLayer = LinearLayer(embedSize, embedSize),
        keyLayer = LinearLayer(embedSize, embedSize),
        valueLayer = LinearLayer(embedSize, embedSize);

  Matrix2d forward(Matrix2d input) {
    var queries = queryLayer.forward(input.toVector()).toMatrix();
    var keys = keyLayer.forward(input.toVector()).toMatrix();
    var values = valueLayer.forward(input.toVector()).toMatrix();

    var scores = (queries * keys.transpose()) / sqrt(input.cols().toDouble());
    var attention = softmax(scores);
    return attention * values;
  }
}

class CrossAttention {
  final LinearLayer queryLayer;
  final LinearLayer keyLayer;
  final LinearLayer valueLayer;

  CrossAttention(int embedSize)
      : queryLayer = LinearLayer(embedSize, embedSize),
        keyLayer = LinearLayer(embedSize, embedSize),
        valueLayer = LinearLayer(embedSize, embedSize);

  Matrix2d forward(Matrix2d encoderOutput, Matrix2d decoderInput) {
    var queries = queryLayer.forward(decoderInput.toVector()).toMatrix();
    var keys = keyLayer.forward(encoderOutput.toVector()).toMatrix();
    var values = valueLayer.forward(encoderOutput.toVector()).toMatrix();

    var scores = (queries * keys.transpose()) / sqrt(encoderOutput.cols().toDouble());
    var attention = softmax(scores);
    return attention * values;
  }
}

Matrix2d softmax(Matrix2d scores) {
  for (int i = 0; i < scores.rows(); i++) {
    double sumExp = scores.row(i).map((v) => exp(v.value)).reduce((a, b) => a + b);
    for (int j = 0; j < scores.cols(); j++) {
      scores.data!.values[i * scores.cols() + j] = Value(exp(scores.at(i, j).value) / sumExp);
    }
  }
  return scores;
}

Matrix2d concatenate(List<Matrix2d> matrices) {
  int totalCols = matrices.fold(0, (sum, mat) => sum + mat.cols());
  Matrix2d result = Matrix2d(matrices[0].rows(), totalCols);

  int colOffset = 0;
  for (var mat in matrices) {
    for (int i = 0; i < mat.rows(); i++) {
      for (int j = 0; j < mat.cols(); j++) {
        result.data!.values[i * totalCols + colOffset + j] = mat.at(i, j);
      }
    }
    colOffset += mat.cols();
  }
  return result;
}

class TransformerBlock {
  final MultiHeadAttention selfAttention;
  final MultiHeadAttention crossAttention;
  final LinearLayer feedForward;

  TransformerBlock(int embedSize, int numHeads)
      : selfAttention = MultiHeadAttention(embedSize, numHeads),
        crossAttention = MultiHeadAttention(embedSize, numHeads),
        feedForward = LinearLayer(embedSize, embedSize);

  Matrix2d forward(Matrix2d input, Matrix2d encoderOutput) {
    var attended = selfAttention.forward(input);
    var crossAttended = crossAttention.forward(encoderOutput);
    return feedForward.forward(crossAttended.toVector()).toMatrix();
  }
}

class Transformer {
  final List<TransformerBlock> encoderBlocks;
  final List<TransformerBlock> decoderBlocks;

  Transformer(int numLayers, int embedSize, int numHeads)
      : encoderBlocks = List.generate(numLayers, (_) => TransformerBlock(embedSize, numHeads)),
        decoderBlocks = List.generate(numLayers, (_) => TransformerBlock(embedSize, numHeads));

  Matrix2d encode(Matrix2d input) {
    Matrix2d output = input;
    for (var block in encoderBlocks) {
      output = block.forward(output, output);
    }
    return output;
  }

  Matrix2d decode(Matrix2d input, Matrix2d encoderOutput) {
    Matrix2d output = input;
    for (var block in decoderBlocks) {
      output = block.forward(output, encoderOutput);
    }
    return output;
  }
}

void main() {
  final transformer = Transformer(2, 5, 4);
  final input = Matrix2d(3, 5);
  final encoded = transformer.encode(input);
  final decoded = transformer.decode(input, encoded);

  print("Decoded Output: ${decoded}");
}
