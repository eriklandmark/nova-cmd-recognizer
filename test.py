import tensorflow as tf
import data_generator
import matplotlib.pyplot as plt
import numpy as np

audio_binary = tf.io.read_file("output.wav")
audio, sample_rate = tf.audio.decode_wav(audio_binary)

print(audio.shape, sample_rate)
first_chanel = np.array([sample[0] for sample in audio])
print(first_chanel.shape)
print(first_chanel)

spect = data_generator.get_spectrogram(first_chanel[:16000], sample_rate=sample_rate)
print(spect.shape)

plt.imshow(spect)

plt.show()
