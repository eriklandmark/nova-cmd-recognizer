import tensorflow as tf
import pyaudio
import numpy as np
import data_generator
import cv2

p = pyaudio.PyAudio()

RATE = 16000

interpreter = tf.lite.Interpreter(model_path="quantized_model/model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

stream = p.open(format=pyaudio.paFloat32, channels=1, rate=RATE, input=True)
stream.start_stream()

labels = data_generator.get_labels()

def quantize(detail, data):
    shape = detail['shape']
    dtype = detail['dtype']
    a, b = detail['quantization']

    return (data.numpy()/a + b).astype(dtype).reshape(shape)

while stream.is_active():
    data = np.frombuffer(stream.read(int(RATE*2)), np.float32)
    spect = data_generator.get_spectrogram(data)
    quantized_input = quantize(input_details[0], spect)
    interpreter.set_tensor(input_details[0]['index'], quantized_input)
    interpreter.invoke()
    quantized_output = interpreter.get_tensor(output_details[0]['index'])
    pred_index = tf.math.argmax(quantized_output[0]).numpy()
    image = tf.cast(tf.math.multiply(spect, 255.), "uint8").numpy()[0]
    cv2.imshow('Frame', image)
    print(labels[pred_index], quantized_output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

print("Stream is stopped")
stream.stop_stream()
stream.close()
p.terminate()
