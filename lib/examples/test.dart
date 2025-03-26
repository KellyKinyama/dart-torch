import '../auto_diff2.dart';

void main() {
  print('--- Example 1: Basic Arithmetic ---');
  final a = Value(2.0);
  final b = Value(3.0);
  final c = a + b;
  c.backward();
  print('c: $c  // Expected: data=5.0');
  print('a: $a  // Expected grad=1.0');
  print('b: $b  // Expected grad=1.0');

  print('\n--- Example 2: Multiplication ---');
  final d = a * b;
  d.backward();
  print('d: $d  // Expected: data=6.0');
  print('a: $a  // Expected grad=3.0');
  print('b: $b  // Expected grad=2.0');

  print('\n--- Example 3: Polynomial y = x^2 + 3x + 1 ---');
  final x1 = Value(2.0);
  final y1 = x1 * x1 + x1 * 3.0 + 1;
  y1.backward();
  print('y1: $y1  // Expected data=11.0');
  print('x1: $x1  // Expected grad=7.0');

  print('\n--- Example 4: Power y = x^3 ---');
  final x2 = Value(2.0);
  final y2 = x2.pow(3);
  y2.backward();
  print('y2: $y2  // Expected data=8.0');
  print('x2: $x2  // Expected grad=12.0');

  print('\n--- Example 5: Negative and Division y = -a / b ---');
  final a2 = Value(4.0);
  final b2 = Value(2.0);
  final y3 = -a2 / b2;
  y3.backward();
  print('y3: $y3  // Expected data=-2.0');
  print('a2: $a2  // Expected grad=-0.5');
  print('b2: $b2  // Expected grad=1.0');

  print('\n--- Example 6: Sigmoid Activation ---');
  final x3 = Value(1.0);
  final y4 = x3.sigmoid();
  y4.backward();
  print('y4: $y4  // Expected ≈ 0.7311');
  print('x3: $x3  // Expected grad ≈ 0.1966');

  print('\n--- Example 7: ReLU Activation (x < 0) ---');
  final x4 = Value(-2.0);
  final y5 = x4.relu();
  y5.backward();
  print('y5: $y5  // Expected data=0.0');
  print('x4: $x4  // Expected grad=0.0');

  print('\n--- Example 8: ReLU Activation (x > 0) ---');
  final x5 = Value(3.0);
  final y6 = x5.relu();
  y6.backward();
  print('y6: $y6  // Expected data=3.0');
  print('x5: $x5  // Expected grad=1.0');

  print('\n--- Example 9: Composite Expression y = sigmoid(a * x + b) * c ---');
  final xc = Value(2.0);
  final ac = Value(3.0);
  final bc = Value(1.0);
  final cc = Value(-1.0);
  final yc = ((ac * xc + bc).sigmoid()) * cc;
  yc.backward();
  print('yc: $yc');
  print('xc: $xc  // Expected small negative grad ≈ -0.00273');
  print('ac: $ac  // Expected ≈ -0.00182');
  print('bc: $bc  // Expected ≈ -0.00091');
  print('cc: $cc  // Expected ≈ sigmoid ≈ 0.99909');

  print('\n--- Example 10: Quadratic Loss = (yTrue - yPred)^2 ---');
  final x6 = Value(2.0);
  final w = Value(3.0);
  final yPred = w * x6;
  final yTrue = Value(10.0);
  final loss = (yTrue - yPred).pow(2);
  loss.backward();
  print('loss: $loss  // Expected=16.0');
  print('x6: $x6  // Expected grad = -24');
  print('w : $w  // Expected grad = -16');
  print('yPred: $yPred');
  print('yTrue: $yTrue');

  print('\n--- Example 11: Chain Rule ---');
  final x7 = Value(2.0);
  final y7 = x7 * 3.0;
  final z7 = y7 + 5.0;
  final out7 = z7.pow(2);
  out7.backward();
  print('out7: $out7  // Expected=121.0');
  print('x7: $x7  // Expected grad=66.0');
}
