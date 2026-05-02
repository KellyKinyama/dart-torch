import 'dart:io';

import 'tokenizer.dart';

class LibriSpeechDataset {
  final String rootDir;
  final EnglishCharacterTokenizer tokenizer;
  final List<Map<String, String>> _samples = [];

  LibriSpeechDataset(this.rootDir, this.tokenizer) {
    _indexDataset();
  }

  void _indexDataset() {
    final root = Directory(rootDir);
    // Traverse: speaker/chapter/*.trans.txt
    root.listSync(recursive: true).forEach((entity) {
      if (entity is File && entity.path.endsWith('.trans.txt')) {
        final lines = entity.readAsLinesSync();
        for (var line in lines) {
          final parts = line.split(' ');
          final id = parts[0];
          final text = parts.sublist(1).join(' ');
          
          // Construct the actual wav/flac path
          final folder = entity.parent.path;
          final audioPath = "$folder/$id.flac"; // Or .wav if converted
          
          if (File(audioPath).existsSync()) {
            _samples.add({'path': audioPath, 'text': text});
          }
        }
      }
    });
    print("Indexed ${_samples.length} samples.");
  }

  Iterable<Map<String, dynamic>> stream(int maxTextLen, int maxAudioLen) sync* {
    for (var sample in _samples) {
      yield {
        'path': sample['path'],
        'text': sample['text'],
      };
    }
  }
}