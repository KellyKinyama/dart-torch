import 'dart:math' as math;
import 'dart:typed_data';

class HarmonicVocoder {
  final int frameSize;
  final int hopSize;
  final int sampleRate;

  HarmonicVocoder({
    this.frameSize = 1024,
    this.hopSize = 256,
    this.sampleRate = 16000,
  });

  Uint8List generate(List<Float64List> mags) {
    final int numFrames = mags.length;
    final int totalLength = numFrames * hopSize + frameSize;

    final signal = Float64List(totalLength);
    final window = _hann(frameSize);

    int bins = mags[0].length;

    // ✅ CRITICAL FIX: persistent phase per frequency bin
    final List<double> phases = List.filled(bins, 0.0);

    for (int f = 0; f < numFrames; f++) {
      int start = f * hopSize;

      for (int n = 0; n < frameSize; n++) {
        double sample = 0.0;

        for (int k = 1; k < bins; k++) {
          // Skip Nyquist overflow
          double freq = k * sampleRate / frameSize;
          if (freq > sampleRate / 2) continue;

          // ✅ CONTINUOUS phase update (NOT reset per frame!)
          phases[k] += 2 * math.pi * freq / sampleRate;

          // ✅ harmonic synthesis
          sample += mags[f][k] * math.sin(phases[k]);
        }

        signal[start + n] += sample * window[n];
      }
    }

    return _toWav(signal);
  }

  Uint8List _toWav(Float64List signal) {
    double maxVal = 0.0;
    for (var s in signal) {
      if (s.abs() > maxVal) maxVal = s.abs();
    }

    final pcm = Int16List(signal.length);

    if (maxVal > 0) {
      for (int i = 0; i < signal.length; i++) {
        pcm[i] = ((signal[i] / maxVal) * 32767).toInt();
      }
    }

    return Uint8List.view(pcm.buffer);
  }

  Float64List _hann(int size) {
    return Float64List.fromList(List.generate(
      size,
      (n) => 0.5 * (1 - math.cos(2 * math.pi * n / (size - 1))),
    ));
  }
}