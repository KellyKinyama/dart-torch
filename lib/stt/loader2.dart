import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:audio_codec/audio_codec.dart';

import 'loader.dart'; // From your discovery

Map<String, String> loadChapterTranscripts(Directory chapterDir) {
  try {
    final transFile = chapterDir
        .listSync()
        .whereType<File>()
        .firstWhere((f) => f.path.endsWith('.trans.txt'));

    final lines = transFile.readAsLinesSync();
    final Map<String, String> transcriptMap = {};

    for (var line in lines) {
      // Format: 103-1240-0000 THE TEXT CONTENT
      final firstSpace = line.indexOf(' ');
      final id = line.substring(0, firstSpace);
      final text = line.substring(firstSpace + 1);
      transcriptMap[id] = text;
    }
    return transcriptMap;
  } catch (e) {
    return {};
  }
}

void main() async {
  // Path updated to your specific download location
  final rootDir = Directory(
      'C:/Users/kkinyama/Downloads/train-clean-100/LibriSpeech/train-clean-100/');
  final processor = SpectroGrammer(sampleRate: 16000, nMels: 32);

  print("Checking directory: ${rootDir.path}");

  if (!await rootDir.exists()) {
    print("Error: Directory not found at ${rootDir.path}");
    // Fallback check: Print contents of the parent if it fails
    final parent = rootDir.parent;
    if (await parent.exists()) {
      print(
          "Parent directory contains: ${parent.listSync().map((e) => e.path.split(Platform.pathSeparator).last).toList()}");
    }
    return;
  }

  print("Starting search for FLAC files...");

  // Iterate through Reader IDs (e.g., 19, 26...)
  await for (var reader in rootDir.list()) {
    if (reader is Directory) {
      // Iterate through Chapter IDs (e.g., 198, 227...)
      await for (var chapter in reader.list()) {
        if (chapter is Directory) {
          final files =
              chapter.listSync().where((f) => f.path.endsWith('.flac'));

          for (var file in files) {
            if (file is! File) continue;

            print(
                "Processing: ${file.path.split(Platform.pathSeparator).last}");

            try {
              // 1. Decode FLAC using audio_codec
              final decoder = FlacDecoder(track: file);
              final result = decoder.decode();

              // LibriSpeech is mono (1 channel), 16-bit PCM
              final totalSamples = result.streamInfoBlock!.totalSamples;
              final pcmInt32 = Int32List(totalSamples);
              int offset = 0;

              while (decoder.hasNextFrame()) {
                final frame = decoder.readFrame();
                final Int32List currentSubframe = frame.subframes[0];

                final int samplesToCopy = currentSubframe.length;

                // Use setRange for high-performance, bounds-checked copying
                if (offset + samplesToCopy <= pcmInt32.length) {
                  pcmInt32.setRange(
                      offset, offset + samplesToCopy, currentSubframe);
                  offset += samplesToCopy;
                } else {
                  // Handle the tail end if the file is slightly longer than totalSamples reported
                  final int remaining = pcmInt32.length - offset;
                  if (remaining > 0) {
                    pcmInt32.setRange(
                        offset, offset + remaining, currentSubframe, 0);
                    offset += remaining;
                  }
                  break;
                }
              }
              decoder.close();

              // 2. Normalize to Float64 (-1.0 to 1.0)
              final doubleSamples = Float64List(pcmInt32.length);
              for (int i = 0; i < pcmInt32.length; i++) {
                // 16-bit normalization
                doubleSamples[i] = pcmInt32[i] / 32768.0;
              }

              // 3. Generate Spectrogram
              final melSpectrogram = processor.compute(doubleSamples);

              print(
                  "Success: 32 Mels x ${melSpectrogram[0].length} Frames extracted.");

              // --- PIVOT: READY FOR TRAINING ---
              // yeah, this is where you feed the melSpectrogram to your AudioTransformer
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
