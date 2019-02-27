import 'dart:math';
import 'dart:io';
import 'dart:typed_data';

// NOTE: Dart's SIMD only supports 4x4 matrices, so if you have more than
// 4 parameters (including the bias, if any), things will need to change.
//
// The point here is basically just to use SIMD to speed up the training process.

void main() {
  var rnd = Random();
  var alpha = Float32x4.splat(0.01);

  // Our equation is z = 3x + 5y + 2
  var inputs = Float32x4List.fromList([
    Float32x4(0, 3, 2, 0),
    Float32x4(1, 4, 2, 0),
    Float32x4(2, 5, 2, 0),
    Float32x4(3, 6, 2, 0),
  ]);

  var outputs = Float32x4List.fromList(
      [17, 25, 33, 41].map((i) => Float32x4.splat(i.toDouble())).toList());

  var weights =
      Float32x4(rnd.nextDouble(), rnd.nextDouble(), rnd.nextDouble(), 0);

  for (var epoch = 0; epoch < 100; epoch++) {
    for (int i = 0; i < inputs.length; i++) {
      var input = inputs[i];
      var expected = outputs[i];
      var weighted = input * weights;
      var computed = Float32x4.splat(weighted.x + weighted.y + weighted.z);
      var diff = computed - expected;
      var error = diff; // Mean squared error
      weights -= error * alpha * input;
    }
  }

  print('Final weights: $weights');

  while (true) {
    stdout.write("Enter x,y separated by comma: ");
    var values = stdin
        .readLineSync()
        .split(',')
        .where((s) => s.trim().isNotEmpty)
        .map(double.parse)
        .toList();
    var x = values[0];
    var y = values[1];
    var input = Float32x4(x, y, 2, 0);
    var actual = ((3 * x) + (5 * y) + 2);
    var weighted = input * weights;
    var computed4 = Float32x4.splat(weighted.x + weighted.y + weighted.z);
    print(computed4);
    var computed = computed4.y;
    var error = ((actual - computed).abs() / computed) * 100;
    print('Actual ~= ${actual.toStringAsFixed(4)}');
    print('Computed ~= ${computed.toStringAsFixed(4)}');
    print('Error ~= ${error.toStringAsFixed(2)}%');
  }
}
