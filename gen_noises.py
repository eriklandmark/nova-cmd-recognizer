import numpy as np
import scipy
import acoustics

for i in range(1000):
    scipy.io.wavfile.write(f'noises/brown_noise_{i}.wav', 16000,
                           np.array(((acoustics.generator.noise(16000,color='brown')) / 3) * 32767).astype(np.int16))
    scipy.io.wavfile.write(f'noises/white_noise_{i}.wav', 16000,
                           np.array(((acoustics.generator.noise(16000,color='white')) / 3) * 32767).astype(np.int16))
    scipy.io.wavfile.write(f'noises/pink_noise_{i}.wav', 16000,
                           np.array(((acoustics.generator.noise(16000,color='pink')) / 3) * 32767).astype(np.int16))

print("Done")