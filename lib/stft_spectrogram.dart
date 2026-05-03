import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:io';

Future<List<Float64List>> stftSpectrogram(
  String path, {
  int frameSize = 1024,
  int hopSize = 256,
}) async {
  final bytes = await File(path).readAsBytes();
  final samples = Int16List.view(bytes.buffer);

  final signal = samples.map((s) => s / 32768.0).toList();

  final window = List.generate(
    frameSize,
    (n) => 0.5 * (1 - math.cos(2 * math.pi * n / (frameSize - 1))),
  );

  int numFrames = (signal.length - frameSize) ~/ hopSize;
  int bins = frameSize ~/ 2 + 1;

  final spec = <Float64List>[];

  for (int f = 0; f < numFrames; f++) {
    final frame = Float64List(bins);
    final start = f * hopSize;

    for (int k = 0; k < bins; k++) {
      double re = 0.0;
      double im = 0.0;

      for (int n = 0; n < frameSize; n++) {
        if (start + n >= signal.length) break;

        double x = signal[start + n] * window[n];
        double angle = 2 * math.pi * k * n / frameSize;

        re += x * math.cos(angle);
        im -= x * math.sin(angle);
      }

      frame[k] = math.sqrt(re * re + im * im);
    }

    spec.add(frame);
  }

  return spec;
}
