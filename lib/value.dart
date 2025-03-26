class Value {
  num value;
  num gradient = 0;
  List<Value> children = [];
  late Function _backward;
  bool visited = false;

  Value(this.value) {
    children = [this];
    _backward = () {};
  }

  Value operator *(Value other) {
    final out = Value(value * other.value);

    out._backward = () {
      //x=a * b
      //dx/da=b
      //dyx/b=a
      //dy/da=dy/dx*dx/da=
      out.gradient += gradient * other.gradient;
      other.gradient += other.gradient * gradient;
    };
    out.children.add(other);

    return out;
  }

  void backward() {
    Set topology = Set.from(children);

    // gradient = 1;
    _backward.call();
    for (var node in topology) {
      if (!node.visited) {
        node.visited = true;
        node.backward();
      }
    }
  }

  @override
  String toString() {
    return "Value (value: $value, gradient: $gradient)";
  }
}

void main() {
  final x = Value(2);
  final y = x * x;

  y.gradient = 5;

  y.backward();

  print("y: $y");
}
