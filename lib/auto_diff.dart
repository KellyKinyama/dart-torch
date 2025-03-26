class Value {
  num data;
  num grad = 0;
  late Set<Value> _prev;
  Function _backward = () {};
  String op = "";

  Value(this.data, {List<Value>? children, this.op = ""}) {
    _prev = children != null ? children.toSet() : {};
  }

  Value operator +(dynamic other) {
    other = other is num ? Value(other) : other;
    final out = Value(data + other.data, children: [other], op: "+");
    _backward = () {
      grad += out.grad;
      other.grad += out.grad;
    };
    out._backward = _backward;
    return out;
  }

  void backward() {
    List<Value> topo = [];
    Set<Value> visited = {};

    buildTopo(Value v) {
      if (!visited.contains(v)) {
        visited.add(v);
        for (Value child in v._prev) {
          buildTopo(child);
        }
        topo.add(v);
      }
    }

    print("Topology: $topo");

    buildTopo(this);

    // go one variable at a time and apply the chain rule to get its gradient
    grad = 1;
    final reverseOrder = topo.reversed;
    for (Value v in reverseOrder) {
      v._backward();
    }
  }

  @override
  String toString() {
    // TODO: implement toString
    return "Value (data: $data, grad: $grad, prev: $_prev, op: $op)";
  }
}

void main() {
  Value x = Value(1);
  Value y = Value(1);
  Value z = x + y + 2;
  Value err = z + Value(5);
  // z.grad = 5;
  err.backward();
  print("Value: ${err._prev}");
}
