import 'dart:io';
import 'dart:typed_data';

class MNISTDataset {
  late List<Uint8List> images;
  late List<int> labels;

  MNISTDataset(this.images, this.labels);

  static Future<MNISTDataset> load(String imagePath, String labelPath) async {
    final images = await _loadImages(imagePath);
    final labels = await _loadLabels(labelPath);
    return MNISTDataset(images, labels);
  }

  static Future<List<Uint8List>> _loadImages(String path) async {
    final file = await File(path).readAsBytes();
    final buffer = ByteData.sublistView(file);

    // Read metadata
    final magicNumber = buffer.getUint32(0, Endian.big);
    if (magicNumber != 0x00000803) {
      throw Exception('Invalid MNIST image file');
    }

    final numImages = buffer.getUint32(4, Endian.big);
    final numRows = buffer.getUint32(8, Endian.big);
    final numCols = buffer.getUint32(12, Endian.big);
    final imageSize = numRows * numCols;

    print('Loading $numImages images of size $numRows x $numCols');

    // Read image data
    List<Uint8List> images = [];
    int offset = 16;
    for (int i = 0; i < numImages; i++) {
      images.add(Uint8List.sublistView(file, offset, offset + imageSize));
      offset += imageSize;
    }

    return images;
  }

  static Future<List<int>> _loadLabels(String path) async {
    final file = await File(path).readAsBytes();
    final buffer = ByteData.sublistView(file);

    // Read metadata
    final magicNumber = buffer.getUint32(0, Endian.big);
    if (magicNumber != 0x00000801) {
      throw Exception('Invalid MNIST label file');
    }

    final numLabels = buffer.getUint32(4, Endian.big);
    print('Loading $numLabels labels');

    // Read label data
    return List<int>.generate(numLabels, (i) => file[8 + i]);
  }
}

List<Float32List> normalizeImages(List<Uint8List> images) {
  return images.map((img) {
    return Float32List.fromList(img.map((px) => px / 255.0).toList());
  }).toList();
}

void main() async {
  const imagePath = 'lib/loader/mnist/train-images.idx3-ubyte';
  const labelPath = 'lib/loader/mnist/train-labels.idx1-ubyte';

  final mnist = await MNISTDataset.load(imagePath, labelPath);

  print(
      'Loaded ${mnist.images.length} images and ${mnist.labels.length} labels.');

  // Display the first label and first image as a grid
  print('First label: ${mnist.labels[0]}');

  final firstImage = mnist.images[0];
  for (int row = 0; row < 28; row++) {
    for (int col = 0; col < 28; col++) {
      final pixel = firstImage[row * 28 + col];
      stdout.write(pixel > 128 ? '█' : ' ');
    }
    print('');
  }
}
