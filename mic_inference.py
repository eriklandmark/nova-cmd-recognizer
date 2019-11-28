import tensorflow as tf
import pyaudio
import numpy as np
import time
import data_generator
from matplotlib import pyplot as plt
from PIL import Image

p = pyaudio.PyAudio()
tf.compat.v1.enable_eager_execution()

CHANNELS = 1
RATE = 16000

fig = plt.figure()
data = np.zeros((118, 128))
plt.ion()

def callback(in_data, frame_count, time_info, flag):
    conv_data = tf.audio.decode_wav(in_data)
    #spect = data_generator.get_spectrogram(conv_data)
    #image = tf.image.resize(tf.expand_dims(spect, -1), (118, 128))
    # image = tf.cast(spect, tf.uint8)
    #image = Image.fromarray(image.numpy())
    #image.save(str(time_info["current_time"]) + ".png")

    print(len(conv_data))

    return None, pyaudio.paContinue


stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    time.sleep(20)
    stream.stop_stream()
    print("Stream is stopped")

stream.close()

p.terminate()
