import tensorflow as tf
import model
import data_generator
import os

BATCH_SIZE = 32
EPOCHS = 20
TOTAL_EXAMPLES = 14538
OUTPUT_PATH = "trained_models"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
train_graph = tf.Graph()
train_sess = tf.compat.v1.Session(graph=train_graph, config=config)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.keras.backend.set_session(train_sess)

# with train_graph.as_default():
print("Loading model")
labels = data_generator.get_labels()
model = model.get_model(len(labels))

#tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
#train_sess.run(tf.compat.v1.global_variables_initializer())

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              metrics=['accuracy']
              )

model.summary()

print("Starting Training")
train_generator = data_generator.train_generator(labels, batch_size=BATCH_SIZE)
eval_generator = data_generator.eval_generator(labels, batch_size=BATCH_SIZE)

model.fit_generator(train_generator, validation_data=eval_generator, steps_per_epoch=int(TOTAL_EXAMPLES / BATCH_SIZE),
                    validation_steps=16, epochs=EPOCHS, verbose=1, workers=20,
                    use_multiprocessing=False, callbacks=[
        #tf.keras.callbacks.TensorBoard(log_dir="tensorboard", write_graph=False, update_freq=100),
        # tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH, "model.{epoch:02d}.ckpt"), save_weights_only=True,
        #                                   verbose=1)
    ])

# model.save(os.path.join(OUTPUT_PATH, "model.h5"))
saver = tf.train.Saver()
saver.save(train_sess, 'checkpoints')
