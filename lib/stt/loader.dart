import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:fftea/fftea.dart';
import 'package:audio_codec/audio_codec.dart'; // Using the library you found

/// --- PART 1: THE REUSABLE SPECTROGRAM GENERATOR ---

class SpectroGrammer {
  final int sampleRate;
  final int nFft;
  final int nMels;
  late List<Float64List> _filterbank;
  late STFT _stft;

  SpectroGrammer({
    this.sampleRate = 16000,
    this.nFft = 2048,
    this.nMels = 32,
  }) {
    final window = Float64List(nFft);
    for (int i = 0; i < nFft; i++) {
      window[i] = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / nFft));
    }
    _stft = STFT(nFft, window);
    _filterbank = _createMelFilterbank();
  }

  List<Float64List> compute(Float64List samples) {
    final hopSize = nFft ~/ 4;
    final List<Float64List> powerSpectrogram = [];

    _stft.run(samples, (Float64x2List freq) {
      final mags = freq.discardConjugates().magnitudes();
      for (int i = 0; i < mags.length; i++) {
        mags[i] = mags[i] * mags[i]; 
      }
      powerSpectrogram.add(mags);
    }, hopSize);

    return _powerToDb(_applyFilterbank(powerSpectrogram));
  }

  List<Float64List> _createMelFilterbank() {
    final nFftBins = nFft ~/ 2 + 1;
    final lowMel = _hzToMel(0.0);
    final highMel = _hzToMel(sampleRate / 2.0);

    final melPoints = List.generate(nMels + 2, 
        (i) => lowMel + i * (highMel - lowMel) / (nMels + 1));
    final hzPoints = melPoints.map(_melToHz).toList();

    final fbank = List.generate(nMels, (_) => Float64List(nFftBins));
    for (int m = 0; m < nMels; m++) {
      final lower = hzPoints[m];
      final center = hzPoints[m + 1];
      final upper = hzPoints[m + 2];
      final norm = 2.0 / (upper - lower);

      for (int i = 0; i < nFftBins; i++) {
        final freq = i * sampleRate / nFft;
        if (freq >= lower && freq <= center) {
          fbank[m][i] = ((freq - lower) / (center - lower)) * norm;
        } else if (freq > center && freq <= upper) {
          fbank[m][i] = ((upper - freq) / (upper - center)) * norm;
        }
      }
    }
    return fbank;
  }

  List<Float64List> _applyFilterbank(List<Float64List> spec) {
    final nFrames = spec.length;
    final result = List.generate(nMels, (_) => Float64List(nFrames));
    for (int m = 0; m < nMels; m++) {
      for (int t = 0; t < nFrames; t++) {
        double sum = 0.0;
        for (int k = 0; k < spec[t].length; k++) {
          sum += spec[t][k] * _filterbank[m][k];
        }
        result[m][t] = sum;
      }
    }
    return result;
  }

  List<Float64List> _powerToDb(List<Float64List> melSpec) {
    return melSpec.map((row) {
      final dbRow = Float64List(row.length);
      for (int i = 0; i < row.length; i++) {
        final val = row[i] < 1e-10 ? 1e-10 : row[i];
        dbRow[i] = 10.0 * (math.log(val) / math.ln10);
      }
      return dbRow;
    }).toList();
  }

  double _hzToMel(double hz) => 2595.0 * (math.log(1.0 + hz / 700.0) / math.ln10);
  double _melToHz(double mel) => 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0);
}

/// --- PART 2: THE DIRECTORY LOADER & RUNNER ---

void main() async {
  final rootDir = Directory('LIBRISPEECH/train-clean-100');
  final processor = SpectroGrammer(sampleRate: 16000, nMels: 32);

  if (!await rootDir.exists()) {
    print("Error: Directory not found at ${rootDir.path}");
    return;
  }

  // Iterate through Reader -> Chapter -> FLAC
  await for (var reader in rootDir.list()) {
    if (reader is Directory) {
      await for (var chapter in reader.list()) {
        if (chapter is Directory) {
          final files = chapter.listSync().where((f) => f.path.endsWith('.flac'));

          for (var file in files) {
            print("Processing: ${file.path}");
            
            // 1. Decode FLAC using audio_codec
            final decoder = FlacDecoder(track: file as File);
            final result = decoder.decode();
            
            // LibriSpeech is mono, so totalSamples is our length
            final pcmInt32 = Int32List(result.streamInfoBlock!.totalSamples);
            int offset = 0;

            while (decoder.hasNextFrame()) {
              final frame = decoder.readFrame();
              // Extract subframe data (LibriSpeech uses 16-bit PCM)
              for (var sample in frame.subframes[0].samples) {
                if (offset < pcmInt32.length) {
                  pcmInt32[offset++] = sample;
                }
              }
            }
            decoder.close();

            // 2. Normalize to Float64 (-1.0 to 1.0)
            final doubleSamples = Float64List(pcmInt32.length);
            for (int i = 0; i < pcmInt32.length; i++) {
              doubleSamples[i] = pcmInt32[i] / 32768.0;
            }

            // 3. Generate Spectrogram
            final melSpectrogram = processor.compute(doubleSamples);

            print("Created Feature Map: 32 Mels x ${melSpectrogram[0].length} Frames");
            
            // --- PIVOT: READY FOR TRAINING ---
            // yeah, here you would pass melSpectrogram to your AudioTransformer
          }
        }
      }
    }
  }
}