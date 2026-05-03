// TODO Implement this library.
import 'dart:math' as math;
import 'dart:typed_data';

class PitchVocoder {
  final int frameSize;
  final int hopSize;

  PitchVocoder({
    this.frameSize = 1024,
    this.hopSize = 256,
  });

  Uint8List generate(List<Float64List> mags, int sampleRate) {
    final rand = math.Random();
    final signal = Float64List(mags.length * hopSize + frameSize);

    final window = _hann(frameSize);

    double phase = 0.0;
    double baseFreq = 120.0; // ✅ default speech pitch

    for (int f = 0; f < mags.length; f++) {
      final start = f * hopSize;

      final frame = Float64List(frameSize);

      for (int i = 0; i < frameSize; i++) {
        // ✅ periodic excitation (voice)
        double voiced = math.sin(phase);
        phase += 2 * math.pi * baseFreq / sampleRate;

        // ✅ noise (unvoiced)
        double noise = (rand.nextDouble() * 2 - 1) * 0.3;

        frame[i] = voiced + noise;
      }

      final shaped = _applyEnvelope(frame, mags[f]);

      for (int i = 0; i < frameSize; i++) {
        signal[start + i] += shaped[i] * window[i];
      }
    }

    return _toWav(signal);
  }

  Float64List _applyEnvelope(Float64List input, Float64List mags) {
    final output = Float64List(input.length);
    int bins = mags.length;

    for (int n = 0; n < input.length; n++) {
      double sample = 0.0;

      for (int k = 0; k < bins; k++) {
        double angle = 2 * math.pi * k * n / input.length;
        sample += mags[k] * math.cos(angle);
      }

      output[n] = sample * input[n];
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
