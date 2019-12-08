import tensorflow as tf
import model
import data_generator
import os

BATCH_SIZE = 32
EPOCHS = 20
TOTAL_EXAMPLES = 14538
OUTPUT_PATH = "trained_models"

# tf.config.set_soft_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# tf.config.threading.set_inter_op_parallelism_threads(4)
# tf.config.threading.set_intra_op_parallelism_threads(4)
# tf.autograph.set_verbosity(2)

print("Loading model")
labels = data_generator.get_labels()
model = model.get_model(len(labels))
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy']
              )
model.summary()

print("Starting Training")
train_generator = data_generator.train_generator(labels, batch_size=BATCH_SIZE, use_images=True)
eval_generator = data_generator.eval_generator(labels, batch_size=BATCH_SIZE, use_images=True)

model.fit_generator(train_generator, validation_data=eval_generator, steps_per_epoch=int(TOTAL_EXAMPLES / BATCH_SIZE),
                    validation_steps=16, epochs=EPOCHS, verbose=1, workers=20,
                    use_multiprocessing=False, callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir="tensorboard", write_graph=False, update_freq=100),
        # tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH, "model.{epoch:02d}.ckpt"), save_weights_only=True,
        #                                   verbose=1)
    ])

model.save(os.path.join(OUTPUT_PATH, "model.h5"))