// In Dart, streams are used to handle asynchronous events, making them ideal for handling real-time data, network responses, user input events, or any data that arrives over time. Dart provides Stream and StreamController classes to work with streams.

// 1. Creating a Simple Stream
// You can create a stream using a StreamController and listen to its events.

import 'dart:async';

void main() {
  // Create a StreamController
  final controller = StreamController<int>();

  // Listen to the stream
  controller.stream.listen(
    (data) => print('Received: $data'),
    onError: (err) => print('Error: $err'),
    onDone: () => print('Stream closed'),
  );

  // Add data to the stream
  controller.add(1);
  controller.add(2);
  controller.add(3);

  // Close the stream
  controller.close();
}
// 2. Using an Asynchronous Stream
// Dart allows you to create asynchronous generators using async* and yield.

Stream<int> countStream(int max) async* {
  for (int i = 1; i <= max; i++) {
    yield i;
    await Future.delayed(Duration(seconds: 1)); // Simulate delay
  }
}

void main() async {
  await for (var value in countStream(5)) {
    print('Received: $value');
  }
}
// 3. Transforming a Stream
// Dart provides map(), where(), and expand() to transform stream data.

// import 'dart:async';

void main() {
  final controller = StreamController<int>();

  // Transforming the stream
  final transformedStream = controller.stream.map((data) => 'Value: $data');

  transformedStream.listen((data) => print(data));

  controller.add(10);
  controller.add(20);
  controller.add(30);

  controller.close();
}
// 4. Handling Errors in Streams
// You can handle errors inside a stream listener.

// import 'dart:async';

void main() {
  final controller = StreamController<int>();

  controller.stream.listen(
    (data) => print('Data: $data'),
    onError: (err) => print('Error: $err'),
    onDone: () => print('Stream closed'),
  );

  controller.add(100);
  controller.addError('Something went wrong');
  controller.add(200);

  controller.close();
}
// 5. Broadcasting a Stream
// By default, a stream is single-subscription (only one listener). You can make it broadcast to allow multiple listeners.


// import 'dart:async';

void main() {
  final controller = StreamController<int>.broadcast();

  controller.stream.listen((data) => print('Listener 1: $data'));
  controller.stream.listen((data) => print('Listener 2: $data'));

  controller.add(5);
  controller.add(10);
  
  controller.close();
}
// 6. Stream Subscription Control
// You can pause, resume, or cancel a stream subscription.


// import 'dart:async';

void main() async {
  final controller = StreamController<int>();

  StreamSubscription<int> subscription = controller.stream.listen((data) {
    print('Received: $data');
  });

  controller.add(1);
  controller.add(2);

  await Future.delayed(Duration(seconds: 1));
  subscription.pause(); // Pause stream

  print('Stream paused');

  await Future.delayed(Duration(seconds: 2));
  subscription.resume(); // Resume stream

  print('Stream resumed');

  controller.add(3);
  controller.add(4);

  await Future.delayed(Duration(seconds: 1));
  subscription.cancel(); // Cancel stream

  print('Stream canceled');

  controller.close();
}
// 7. Using Stream.fromIterable()
// A quick way to create a stream from a list.



void main() async {
  Stream<int> numbers = Stream.fromIterable([10, 20, 30, 40]);

  await for (var num in numbers) {
    print('Number: $num');
  }
}
// 8. Merging Two Streams
// You can merge multiple streams using StreamGroup from the async package.

// import 'dart:async';

void main() {
  final stream1 = Stream.periodic(Duration(seconds: 1), (i) => 'A$i').take(3);
  final stream2 = Stream.periodic(Duration(seconds: 1), (i) => 'B$i').take(3);

  final mergedStream = Stream.fromFutures([stream1.first, stream2.first]);

  mergedStream.listen((event) {
    print(event);
  });
}
// Conclusion
// Dart's streams provide a powerful way to work with asynchronous data. You can: ✔ Create streams using StreamController
// ✔ Use async* and yield to generate streams
// ✔ Transform and filter streams
// ✔ Handle errors
// ✔ Broadcast streams to multiple listeners
// ✔ Pause, resume, and cancel subscriptions

// Let me know if you need more details or a specific use case! 🚀