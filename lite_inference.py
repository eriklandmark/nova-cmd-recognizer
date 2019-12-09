import tensorflow as tf
import os

tf.compat.v1.enable_eager_execution()

import data_generator
import timeit

model_path = "quantized_model/model.tflite"


if os.name == 'nt':
    interpreter = tf.lite.Interpreter(model_path)
else:
    import tflite_runtime as tflite
    interpreter = tflite.Interpreter(model_path,
                                        experimental_delegates=[tf.lite.load_delegate('libedgetpu.so.1')])

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

dataset =  data_generator.generate_inf_dataset()

def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']

    return (data/a + b).astype(dtype).reshape(shape)

def eval():
    total_seen = 0
    num_correct = 0
    for (sample, label) in dataset:
        total_seen += 1
        quantized_input = quantize(input_details[0], sample)
        interpreter.set_tensor(input_details[0]['index'], quantized_input)
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_details[0]['index'])
        if tf.math.argmax(quantized_output[0]).numpy() == label:
            num_correct += 1

        if total_seen % 100 == 0:
            print("Accuracy after %i images: %f" %
                  (total_seen, float(num_correct) / float(total_seen)))

    print(f"Total seen: {total_seen}")
    print(f"had {num_correct} rights. ({float(num_correct) / float(total_seen)} %)")


time = timeit.timeit(eval, number=1)
print(f"In {time} seconds")