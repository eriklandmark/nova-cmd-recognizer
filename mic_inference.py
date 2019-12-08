import tensorflow as tf
import pyaudio
import numpy as np
import data_generator
import cv2

p = pyaudio.PyAudio()
tf.compat.v1.enable_eager_execution()

CHANNELS = 1
RATE = 16000

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True)

stream.start_stream()

while stream.is_active():
    data = np.frombuffer(stream.read(int(RATE*2)), np.float32)
    spect = data_generator.get_spectrogram(data)
    cv2.imshow('Frame', spect.numpy())
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    print(data.shape)


stream.stop_stream()

print("Stream is stopped")

stream.close()

p.terminate()
