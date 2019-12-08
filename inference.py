import tensorflow as tf
import model
import data_generator
import os

tf.config.set_soft_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.autograph.set_verbosity(10)

model = tf.keras.models.load_model("trained_models_v3/model.h5")

labels = data_generator.get_labels()
eval_generator = data_generator.eval_generator(labels, batch_size=1, use_images=True)

data = [eval_generator.__getitem__(i) for i in range(eval_generator.__len__())]

total_rights = 0

for [x, y] in data:
    pred_data = model.predict(x)
    answer = tf.argmax(pred_data[0]).numpy()
    if answer == tf.argmax(y[0]).numpy():
        total_rights += 1
        print(f"Got {total_rights} / {len(data)} right!")

print(f"{total_rights / len(data)} % right!")