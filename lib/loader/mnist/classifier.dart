import 'dart:io';
import 'dart:typed_data';
import 'dart:math';

// Assume your MultiLayerPerceptron class is defined elsewhere
import 'mlp.dart';  

class MNISTDataset {
  late List<Float32List> images;
  late List<int> labels;

  MNISTDataset(this.images, this.labels);

  static Future<MNISTDataset> load(String imagePath, String labelPath) async {
    final images = await _loadImages(imagePath);
    final labels = await _loadLabels(labelPath);
    return MNISTDataset(images, labels);
  }

  static Future<List<Float32List>> _loadImages(String path) async {
    final file = await File(path).readAsBytes();
    final buffer = ByteData.sublistView(file);

    // Read metadata
    final numImages = buffer.getUint32(4, Endian.big);
    final numRows = buffer.getUint32(8, Endian.big);
    final numCols = buffer.getUint32(12, Endian.big);
    final imageSize = numRows * numCols;

    print('Loading $numImages images of size $numRows x $numCols');

    // Read and normalize image data
    List<Float32List> images = [];
    int offset = 16;
    for (int i = 0; i < numImages; i++) {
      final pixels = Uint8List.sublistView(file, offset, offset + imageSize);
      images.add(Float32List.fromList(pixels.map((px) => px / 255.0).toList())); // Normalize to [0,1]
      offset += imageSize;
    }
    return images;
  }

  static Future<List<int>> _loadLabels(String path) async {
    final file = await File(path).readAsBytes();
    final buffer = ByteData.sublistView(file);

    final numLabels = buffer.getUint32(4, Endian.big);
    print('Loading $numLabels labels');

    return List<int>.generate(numLabels, (i) => file[8 + i]);
  }
}

void main() async {
  const trainImagePath = 'train-images-idx3-ubyte';
  const trainLabelPath = 'train-labels-idx1-ubyte';

  final mnist = await MNISTDataset.load(trainImagePath, trainLabelPath);
  print('Loaded ${mnist.images.length} images and ${mnist.labels.length} labels.');

  // Initialize Neural Network
  final model = MultiLayerPerceptron(
    inputSize: 784, // 28x28 images flattened
    hiddenSizes: [128, 128], // Two hidden layers
    outputSize: 10, // Digits 0-9
    learningRate: 0.01,
  );

  // Train the model
  trainModel(model, mnist.images, mnist.labels, epochs: 5, batchSize: 32);

  // Test on first image
  final prediction = model.predict(mnist.images[0]);
  print('Predicted digit: ${prediction.indexWhere((v) => v == prediction.reduce(max))}');
}

void trainModel(MultiLayerPerceptron model, List<Float32List> images, List<int> labels,
    {int epochs = 5, int batchSize = 32}) {
  final random = Random();

  for (int epoch = 0; epoch < epochs; epoch++) {
    double totalLoss = 0.0;

    for (int i = 0; i < images.length; i++) {
      final input = images[i];
      final target = List<double>.filled(10, 0.0);
      target[labels[i]] = 1.0; // One-hot encoding

      final loss = model.train(input, target);
      totalLoss += loss;

      if (i % 1000 == 0) print('Epoch $epoch, Sample $i, Loss: ${loss.toStringAsFixed(4)}');
    }

    print('Epoch $epoch complete. Average Loss: ${(totalLoss / images.length).toStringAsFixed(4)}');
  }
}
