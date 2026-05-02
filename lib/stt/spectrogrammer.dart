import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:fftea/fftea.dart';

import '../nn/value_vector.dart';

// Note: These imports assume your project structure or package availability
// import 'package:audio_codec/src/flac/flac_decoder.dart';

/// --- PART 1: THE CORE MATH (Spectro Grammer) ---

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
    // 1. Initialize STFT with a Hann Window
    final window = Float64List(nFft);
    for (int i = 0; i < nFft; i++) {
      window[i] = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / nFft));
    }
    _stft = STFT(nFft, window);

    // 2. Pre-compute Mel Filterbank
    _filterbank = _createMelFilterbank();
  }

  /// Converts a raw PCM list to a Log-Mel Spectrogram
  List<Float64List> compute(Float64List samples) {
    final hopSize = nFft ~/ 4;
    final List<Float64List> powerSpectrogram = [];

    // Run STFT
    _stft.run(samples, (Float64x2List freq) {
      final mags = freq.discardConjugates().magnitudes();
      for (int i = 0; i < mags.length; i++) {
        mags[i] = mags[i] * mags[i]; // Power
      }
      powerSpectrogram.add(mags);
    }, hopSize);

    // Apply Mel Filterbank
    final melSpec = _applyFilterbank(powerSpectrogram);

    // Convert to Decibels
    return _powerToDb(melSpec);
  }

  List<Float64List> _createMelFilterbank() {
    final nFftBins = nFft ~/ 2 + 1;
    final lowMel = _hzToMel(0.0);
    final highMel = _hzToMel(sampleRate / 2.0);

    final melPoints = List.generate(
        nMels + 2, (i) => lowMel + i * (highMel - lowMel) / (nMels + 1));
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
    // Result: nMels rows x nFrames columns
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

  double _hzToMel(double hz) =>
      2595.0 * (math.log(1.0 + hz / 700.0) / math.ln10);
  double _melToHz(double mel) => 700.0 * (math.pow(10.0, mel / 2595.0) - 1.0);
}

/// --- PART 2: THE INTEGRATED DECODER & RUNNER ---

void main() async {
  final flacFile = File('test.flac');
  if (!await flacFile.exists()) {
    print("Please provide a test.flac file.");
    return;
  }

  print("Decoding FLAC...");
  // Using the audio_codec approach you found
  // final decoder = FlacDecoder(track: flacFile);
  // final result = decoder.decode();

  // PLACEHOLDER: Since I cannot run the specific audio_codec binary here,
  // assume we have extracted the PCM samples into this list:
  final Int32List pcmInt32 = Int32List(16000 * 5); // 5 seconds of 16kHz audio

  // 1. Normalize PCM to Double (-1.0 to 1.0)
  final samples = Float64List(pcmInt32.length);
  for (int i = 0; i < pcmInt32.length; i++) {
    // LibriSpeech is 16-bit, so divide by 2^15
    samples[i] = pcmInt32[i] / 32768.0;
  }

  // 2. Initialize Spectro Grammer
  final grammer = SpectroGrammer(
    sampleRate: 16000, // LibriSpeech standard
    nMels: 32, // Matches your AudioTransformer input
  );

  // 3. Compute Features
  print("Computing Mel Spectrogram...");
  final spectrogram = grammer.compute(samples);

  print("Done!");
  print(
      "Spectrogram Shape: ${spectrogram.length} Mels x ${spectrogram[0].length} Frames");

  // This 'spectrogram' can now be converted to ValueVectors for your model
  final inputTensor =
      spectrogram.map((row) => ValueVector.fromDoubleList(row)).toList();
}
