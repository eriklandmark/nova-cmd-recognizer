import tensorflow as tf
import data_generator
import matplotlib.pyplot as plt

audio_binary = tf.io.read_file("output.wav")
audio, sample_rate = tf.audio.decode_wav(audio_binary)

print(audio.shape, sample_rate)


spect = data_generator.get_spectrogram(audio[:8000], sample_rate=sample_rate)
print(spect.shape)

plt.imshow(spect)

plt.show()
