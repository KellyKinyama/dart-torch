class Dual {
  final double value;
  final double derivative;

  Dual(this.value, this.derivative);

  // Addition
  Dual operator +(Dual other) =>
      Dual(value + other.value, derivative + other.derivative);

  // Subtraction
  Dual operator -(Dual other) =>
      Dual(value - other.value, derivative - other.derivative);

  // Multiplication (product rule)
  Dual operator *(Dual other) => Dual(
      value * other.value, value * other.derivative + derivative * other.value);

  // Division (quotient rule)
  Dual operator /(Dual other) {
    final denom = other.value * other.value;
    return Dual(value / other.value,
        (derivative * other.value - value * other.derivative) / denom);
  }

  // Exponential function
  Dual exp() {
    final expVal = value.exp();
    return Dual(expVal, expVal * derivative);
  }

  // Natural logarithm
  Dual log() {
    return Dual(value.log(), derivative / value);
  }

  // Sine function
  Dual sin() {
    return Dual(value.sin(), derivative * value.cos());
  }

  // Cosine function
  Dual cos() {
    return Dual(value.cos(), -derivative * value.sin());
  }

  @override
  String toString() => 'Dual(value: $value, derivative: $derivative)';
}

void main() {
  // Example: Compute derivative of f(x) = x^2 + 3x at x = 2
  Dual x = Dual(2.0, 1.0); // x with derivative 1 (since df/dx = 1)
  Dual f = x * x + x * Dual(3.0, 0.0); // f(x) = x^2 + 3x

  print('f(x) at x=2: ${f.value}'); // Expected: 2^2 + 3(2) = 10
  print('df/dx at x=2: ${f.derivative}'); // Expected: 2(2) + 3 = 7
}
