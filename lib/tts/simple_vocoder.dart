import 'dart:math' as math;
import 'dart:typed_data';

class SimpleVocoder {
  final int frameSize;
  final int hopSize;

  SimpleVocoder({
    this.frameSize = 1024,
    this.hopSize = 256,
  });

  Uint8List generate(List<Float64List> magnitudes, int sampleRate) {
    final rand = math.Random();
    final signal = Float64List(magnitudes.length * hopSize + frameSize);

    final window = _hann(frameSize);

    for (int f = 0; f < magnitudes.length; f++) {
      final start = f * hopSize;

      // Generate noise excitation
      final noise = Float64List(frameSize);
      for (int i = 0; i < frameSize; i++) {
        noise[i] = rand.nextDouble() * 2 - 1;
      }

      // Shape noise using magnitude spectrum
      final shaped = _applySpectralEnvelope(noise, magnitudes[f]);

      // Overlap-add
      for (int i = 0; i < frameSize; i++) {
        signal[start + i] += shaped[i] * window[i];
      }
    }

    return _toWav(signal);
  }

  // ✅ Apply spectral envelope (VERY IMPORTANT)
  Float64List _applySpectralEnvelope(Float64List noise, Float64List mags) {
    final output = Float64List(noise.length);
    int bins = mags.length;

    for (int n = 0; n < noise.length; n++) {
      double sample = 0.0;

      for (int k = 0; k < bins; k++) {
        double angle = 2 * math.pi * k * n / noise.length;
        sample += mags[k] * math.cos(angle);
      }

      // Multiply by noise excitation
      output[n] = sample * noise[n];
    }

    return output;
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
