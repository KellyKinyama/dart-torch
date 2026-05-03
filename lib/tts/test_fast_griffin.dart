import 'dart:io';
import 'dart:typed_data';

import '../stft_spectrogram.dart';
import 'fast_griffin_lim.dart';

Future<void> main() async {
  print('---- TESTING FAST GRIFFIN-LIM ----');

  const sampleRate = 16000;
  const frameSize = 1024;
  const hopSize = 256;
  const maxFrames = 64;

  final spectrogram = await stftSpectrogram(
    'output.wav',
    frameSize: frameSize,
    hopSize: hopSize,
  );

  final magnitudes = spectrogram.take(maxFrames).map((frame) {
    final out = Float64List(frame.length);
    for (int i = 0; i < frame.length; i++) {
      final v = frame[i];
      out[i] = (v.isFinite ? v : 0.0).clamp(1e-8, 1e8);
    }
    return out;
  }).toList();

  print('Frames: ${magnitudes.length}');
  print('Bins:   ${magnitudes.isNotEmpty ? magnitudes[0].length : 0}');

  final fast = FastGriffinLimGenerator(
    iterations: 60,
    frameSize: frameSize,
    hopSize: hopSize,
    momentum: 0.99,
  );

  print('Running Fast Griffin–Lim...');
  final bytes = fast.generateWav(magnitudes, sampleRate);

  final wavBytes = _ensureWavPcm16Mono(bytes, sampleRate);
  await File('test_fast_griffin.wav').writeAsBytes(wavBytes);

  print('\n✅ Saved: test_fast_griffin.wav');
}

Uint8List _ensureWavPcm16Mono(Uint8List bytes, int sampleRate) {
  if (_looksLikeWav(bytes)) return bytes;
  return _wrapRawPcm16leToWav(bytes, sampleRate, channels: 1);
}

bool _looksLikeWav(Uint8List bytes) {
  if (bytes.length < 12) return false;
  return bytes[0] == 0x52 && // R
      bytes[1] == 0x49 && // I
      bytes[2] == 0x46 && // F
      bytes[3] == 0x46 && // F
      bytes[8] == 0x57 && // W
      bytes[9] == 0x41 && // A
      bytes[10] == 0x56 && // V
      bytes[11] == 0x45; // E
}

Uint8List _wrapRawPcm16leToWav(
  Uint8List pcmBytes,
  int sampleRate, {
  int channels = 1,
}) {
  const bitsPerSample = 16;
  final bytesPerSample = bitsPerSample ~/ 8;
  final blockAlign = channels * bytesPerSample;
  final byteRate = sampleRate * blockAlign;

  final dataSize = pcmBytes.length;
  final riffSize = 36 + dataSize;

  final header = ByteData(44);
  void writeAscii(int offset, String s) {
    for (int i = 0; i < s.length; i++) {
      header.setUint8(offset + i, s.codeUnitAt(i));
    }
  }

  writeAscii(0, 'RIFF');
  header.setUint32(4, riffSize, Endian.little);
  writeAscii(8, 'WAVE');

  writeAscii(12, 'fmt ');
  header.setUint32(16, 16, Endian.little);
  header.setUint16(20, 1, Endian.little);
  header.setUint16(22, channels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, byteRate, Endian.little);
  header.setUint16(32, blockAlign, Endian.little);
  header.setUint16(34, bitsPerSample, Endian.little);

  writeAscii(36, 'data');
  header.setUint32(40, dataSize, Endian.little);

  final out = Uint8List(44 + dataSize);
  out.setAll(0, header.buffer.asUint8List());
  out.setAll(44, pcmBytes);
  return out;
}
