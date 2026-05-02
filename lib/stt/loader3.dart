import 'dart:io';
import 'dart:typed_data';
import 'package:audio_codec/audio_codec.dart';

import 'loader.dart'; // From your discovery

/// Parses the .trans.txt file within a chapter directory to map audio IDs to text labels.
Map<String, String> loadChapterTranscripts(Directory chapterDir) {
  try {
    final transFile = chapterDir
        .listSync()
        .whereType<File>()
        .firstWhere((f) => f.path.endsWith('.trans.txt'));

    final lines = transFile.readAsLinesSync();
    final Map<String, String> transcriptMap = {};

    for (var line in lines) {
      // Line format: 103-1240-0000 THE TEXT CONTENT
      final firstSpace = line.indexOf(' ');
      if (firstSpace == -1) continue;

      final id = line.substring(0, firstSpace);
      final text = line.substring(firstSpace + 1);
      transcriptMap[id] = text;
    }
    return transcriptMap;
  } catch (e) {
    // Return empty map if transcript is missing or unreadable
    return {};
  }
}

void main() async {
  // Path to your LibriSpeech download
  final rootDir = Directory(
      'C:/Users/kkinyama/Downloads/train-clean-100/LibriSpeech/train-clean-100/');

  // nMels 32 as per your current pipeline configuration
  final processor = SpectroGrammer(sampleRate: 16000, nMels: 32);

  print("Checking directory: ${rootDir.path}");

  if (!await rootDir.exists()) {
    print("Error: Directory not found at ${rootDir.path}");
    return;
  }

  print("Starting search for FLAC files...");

  // Iterate through Reader IDs (e.g., 103, 1241...)
  await for (var reader in rootDir.list()) {
    if (reader is Directory) {
      // Iterate through Chapter IDs
      await for (var chapter in reader.list()) {
        if (chapter is Directory) {
          // 1. Load all transcripts for this chapter once to avoid redundant I/O
          final transcripts = loadChapterTranscripts(chapter);

          final files = chapter
              .listSync()
              .where((f) => f.path.endsWith('.flac'))
              .whereType<File>();

          for (var file in files) {
            final fileName = file.path.split(Platform.pathSeparator).last;
            final fileId = fileName.replaceAll('.flac', '');
            final textLabel = transcripts[fileId] ?? "UNKNOWN";

            print("Processing: $fileName | Label: $textLabel");

            try {
              // 2. Decode FLAC using audio_codec
              final decoder = FlacDecoder(track: file);
              decoder.decode(); // Initializes headers and metadata

              // DYNAMIC BUFFER: Collect subframes in a list to prevent RangeErrors
              // caused by metadata mismatches in totalSamples.
              final List<Int32List> allSubframes = [];
              int totalCollectedSamples = 0;

              while (decoder.hasNextFrame()) {
                final frame = decoder.readFrame();
                final subframe = frame.subframes[0]; // LibriSpeech is Mono
                allSubframes.add(subframe);
                totalCollectedSamples += subframe.length;
              }
              decoder.close();

              // Flatten collected subframes into a single contiguous buffer
              final pcmInt32 = Int32List(totalCollectedSamples);
              int offset = 0;
              for (final sub in allSubframes) {
                pcmInt32.setRange(offset, offset + sub.length, sub);
                offset += sub.length;
              }

              // 3. Normalize to Float64 (-1.0 to 1.0) for the SpectroGrammer
              final doubleSamples = Float64List(pcmInt32.length);
              for (int i = 0; i < pcmInt32.length; i++) {
                // 16-bit PCM normalization
                doubleSamples[i] = pcmInt32[i] / 32768.0;
              }

              // 4. Generate Mel-Spectrogram Features
              final melSpectrogram = processor.compute(doubleSamples);

              print(
                  "Success: 32 Mels x ${melSpectrogram[0].length} Frames extracted.");

              // --- PIVOT: READY FOR TRAINING ---
              // yeah, you now have:
              // X: melSpectrogram (List<Float64List>)
              // Y: textLabel (String)
              // This is where you pass them to your Stable Tensor-Engine/AudioTransformer.
            } catch (e) {
              print("Failed to process ${file.path}: $e");
            }
          }
        }
      }
    }
  }
  print("Data loading cycle complete.");
}
