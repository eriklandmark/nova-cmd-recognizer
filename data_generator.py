import tensorflow as tf
import os
import random
import numpy as np

DATASET_PATH = "D:\Datasets\TFSpeechRecognition\data"
commands = ["up", "down", "left", "right"]
test_dataset_file = "D:\Datasets\TFSpeechRecognition\\testing_list.txt"
validation_dataset_file = "D:\Datasets\TFSpeechRecognition\\validation_list.txt"
noises_dataset_files = "D:\Datasets\TFSpeechRecognition\data\_background_noise_\\noises"

def get_labels():
    return commands + ["noise"]

def get_spectrogram(audio_signals, sample_rate=16000):
    signals = tf.reshape(audio_signals, [1, -1])
    magnitude_spectrograms = tf.abs(tf.signal.stft(signals, frame_length=1024, frame_step=128, fft_length=1024))
    num_spectrogram_bins = magnitude_spectrograms.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(128, num_spectrogram_bins, sample_rate, 20,
                                                                        8000)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    return log_mel_spectrograms[0]


def process_audio(audio_path):
    audio_binary = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    spect = get_spectrogram(audio, sample_rate=sample_rate)
    image = tf.image.resize(tf.expand_dims(spect, -1), (118, 128))
    image = tf.math.divide(tf.cast(image, "float32"), 255.)
    return image


def train_generator(labels, batch_size=32):
    with open("train_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]
    return DataGenerator(data_ids, labels, **{'batch_size': batch_size})

def eval_generator(labels, batch_size=32):
    with open("test_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]
    return DataGenerator(data_ids, labels, **{'batch_size': batch_size})


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_ids, labels, batch_size=16, shuffle=False, verbose=False):
        self.batch_size = batch_size
        self.labels = labels
        self.data_ids = data_ids
        self.shuffle = shuffle
        self.verbose = verbose

    def __len__(self):
        return int(tf.math.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, batch_id):
        batch = self.data_ids[batch_id * self.batch_size: (batch_id * self.batch_size) + self.batch_size]
        if self.verbose:
            print(f"Creating batch {batch_id}")
        return self.create_batch(batch, self.labels)

    def create_batch(self, data_array, labels, verbose=False):
        examples = []
        computed_labels = []

        for path in data_array:
            example = process_audio(os.path.join(DATASET_PATH, path.strip()))
            np_example = np.asarray(example)
            examples.append(np_example)
            label = "noise" if path.startswith("_background") else path.split("\\")[0]
            computed_labels.append(labels.index(label))

        categorical_labels = np.asarray(
           [tf.keras.utils.to_categorical(label, num_classes=len(labels), dtype='float32') for label in computed_labels])

        return np.asarray(examples), categorical_labels


def Diff(li1, li2):
    return (list(set(li1) - set(li2)))


def gen_datasets_lists():
    all_samples = []

    test_samples = []

    with open(test_dataset_file, "r") as t_file:
        test_file_lines = [line.strip() for line in t_file.readlines()]

    with open(validation_dataset_file, "r") as v_file:
        eval_file_lines = [line.strip() for line in v_file.readlines()]

    for cmd in commands:
        all_samples.extend([cmd + "/" + file for file in os.listdir(os.path.join(DATASET_PATH, cmd))])

        for test_sample in test_file_lines:
            if test_sample.startswith(cmd): test_samples.append(test_sample)

        for eval_sample in eval_file_lines:
            if eval_sample.startswith(cmd): test_samples.append(eval_sample)


    noises = ["_background_noise_/noises/" + file for file in os.listdir(noises_dataset_files)]
    train_noises = noises[0:int(len(noises) * 0.8)]
    test_noises = Diff(noises, train_noises)
    train_files = Diff(all_samples, test_samples) + train_noises
    test_files = test_samples + test_noises

    print(len(train_files))
    print(len(test_noises))

    random.shuffle(train_files)
    random.shuffle(test_files)

    with open("train_list.txt", "w") as train_file:
        for tr_file in train_files:
            train_file.write(tr_file.replace("/", "\\") + "\n")

    with open("test_list.txt", "w") as test_file:
        for tr_file in test_files:
            test_file.write(tr_file.replace("/", "\\") + "\n")

if __name__ == "__main__":
    gen_datasets_lists()