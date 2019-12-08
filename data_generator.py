import tensorflow as tf
import os
import random
import numpy as np
import math
import lib
import cv2

DATASET_PATH = "D:\Datasets\TFSpeechRecognition\data"
IMAGE_DATASET_PATH = "train_images"
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
                                                                        sample_rate / 2)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    image = tf.image.resize(tf.expand_dims(mel_spectrograms, -1), (118, 128))
    return image


def process_audio(audio_path, convert_to_int=False, from_image=False):
    content = tf.io.read_file(audio_path)

    if from_image:
        image = tf.io.decode_png(content)
        if not convert_to_int:
            image = tf.math.divide(tf.cast(image, "float32"), 255.)
    else:
        audio, sample_rate = tf.audio.decode_wav(content)
        image = get_spectrogram(audio, sample_rate)
        if convert_to_int:
            image = tf.cast(tf.math.multiply(image, 255.), "uint8")
    return image


def train_generator(labels, batch_size=32, use_images=False):
    with open("train_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]
    return DataGenerator(data_ids, labels, **{'batch_size': batch_size}, verbose=False, use_images=use_images)


def eval_generator(labels, batch_size=32, use_images=False):
    with open("test_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]
    return DataGenerator(data_ids, labels, **{'batch_size': batch_size}, use_images=use_images)


def representative_data_gen():
    with open("train_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]
    random.shuffle(data_ids)
    for path in data_ids[:100]:
        example = process_audio(os.path.join(DATASET_PATH, path.strip()))
        np_example = np.asarray(example)
        yield [np_example]


def compute_label(label_path, labels=None):
    if not labels:
        labels = get_labels()

    label = "noise" if label_path.startswith("_background") else label_path.split("\\")[0]
    return labels.index(label)


def generate_dataset_with_batch(batch_size=32, mode="train", use_images=False):
    x = []
    y = []

    with open(f"{mode}_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]

    for batch_id in range(int(len(data_ids) / batch_size)):
        batch_ids = data_ids[batch_id * batch_size: (batch_id * batch_size) + batch_size]
        batch_x = []
        batch_y = []
        for i, path in enumerate(batch_ids):
            if use_images:
                pre, ext = os.path.splitext(path)
                example = process_audio(os.path.join(IMAGE_DATASET_PATH, pre + ".png"), from_image=True)
            else:
                example = process_audio(os.path.join(DATASET_PATH, path.strip()))
            batch_x.append(np.asarray(example))
            batch_y.append()
            step = (batch_id * batch_size) + i + 1
            lib.progress_bar(step, len(data_ids), prefix="Creating examples",
                             suffix=f'Completed ({step}/{len(data_ids)})', length=30)

        x.append(batch_x)
        y.append(batch_y)

    return np.asarray(x), np.asarray(y)


def generate_dataset(labels, mode="train", use_images=False):
    x = []
    y = []

    with open(f"{mode}_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]

    for i, path in enumerate(data_ids):
        if use_images:
            pre, ext = os.path.splitext(path)
            example = process_audio(os.path.join(IMAGE_DATASET_PATH, pre + ".png"), from_image=True)
        else:
            example = process_audio(os.path.join(DATASET_PATH, path.strip()))

        x.append(np.asarray(example))
        y.append(compute_label(path))
        lib.progress_bar(i + 1, len(data_ids), prefix="Creating examples",
                         suffix=f'Completed ({i + 1}/{len(data_ids)})', length=30)

    return np.asarray(x), np.asarray(
        [tf.keras.utils.to_categorical(label, num_classes=len(labels), dtype='float32') for label in y])


def generate_inf_dataset():
    with open("test_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]

    dataset = []

    for i, path in enumerate(data_ids):
        pre, ext = os.path.splitext(path)
        example = process_audio(os.path.join(IMAGE_DATASET_PATH, pre + ".png"), from_image=True)
        dataset.append((np.asarray(example), compute_label(path)))
        lib.progress_bar(i + 1, len(data_ids), prefix="Creating examples",
                         suffix=f'Completed ({i + 1}/{len(data_ids)})', length=30)

    return dataset


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_ids, labels, batch_size=16, shuffle=False, verbose=False, use_images=False):
        self.batch_size = batch_size
        self.labels = labels
        self.data_ids = data_ids
        self.shuffle = shuffle
        self.verbose = verbose
        self.use_images = use_images

    def __len__(self):
        return int(math.floor(len(self.data_ids) / self.batch_size))

    def __getitem__(self, batch_id):
        batch = self.data_ids[batch_id * self.batch_size: (batch_id * self.batch_size) + self.batch_size]
        if self.verbose:
            print(f"Creating batch {batch_id}")
        return self.create_batch(batch, self.labels)

    def create_batch(self, data_array, labels, verbose=False):
        examples = []
        computed_labels = []

        for i, path in enumerate(data_array):
            if self.use_images:
                pre, ext = os.path.splitext(path)
                example = process_audio(os.path.join(IMAGE_DATASET_PATH, pre + ".png"), from_image=True)
            else:
                example = process_audio(os.path.join(DATASET_PATH, path.strip()))

            np_example = np.asarray(example)
            examples.append(np_example)
            label = compute_label(path, labels)
            computed_labels.append(label)

            if verbose:
                lib.progress_bar(i + 1, len(data_array), prefix="Creating examples",
                                 suffix=f'Completed ({i + 1}/{len(data_array)})', length=30)

        categorical_labels = np.asarray(
            [tf.keras.utils.to_categorical(label, num_classes=len(labels), dtype='float32') for label in
             computed_labels])

        return np.asarray(examples), categorical_labels


def Diff(li1, li2):
    return list(set(li1) - set(li2))


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


def create_train_images():
    with open("test_list.txt", "r") as train_file:
        data_ids = [f.strip() for f in train_file.readlines()]

    for i, path in enumerate(data_ids):
        example = process_audio(os.path.join(DATASET_PATH, path.strip()), convert_to_int=True)
        pre, ext = os.path.splitext(path)
        cv2.imwrite("train_images/" + pre + ".png", example.numpy()[0])
        lib.progress_bar(i + 1, len(data_ids), prefix="Creating examples",
                         suffix=f'Completed ({i + 1}/{len(data_ids)})', length=30)


if __name__ == "__main__":
    # gen_datasets_lists()
    create_train_images()
