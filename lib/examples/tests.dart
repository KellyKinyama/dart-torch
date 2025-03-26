import '../value2.dart';

void testSanityCheck() {
  final x = Value(-4.0);
  final z = Value(2) * x + 2 + x;
  final q = z.relu() + z * x;
  final h = (z * z).relu();
  final y = h + q + q * x;
  y.backward();
}

void main() {
  final a = Value(-4.0);
  final b = Value(2.0);
  Value c = a + b;
  Value d = a * b + b.pow(3);
  c += c + 1;
  c += Value(1) + c + (-a);
  d += d * 2 + (b + a).relu();
  d += Value(3) * d + (b - a).relu();
  final e = c - d;
  final f = e.pow(2);
  Value g = f / 2.0;
  g += Value(10.0) / f;
  print('${g.data}. Expected: 24.7041'); // prints 24.7041, the outcome of this forward pass
  g.backward();
  print('${a.grad}. Expected: 138.8338'); // prints 138.8338, i.e. the numerical value of dg/da
  print('${b.grad}. Expected: 645.5773'); // prints 645.5773, i.e. the
}
