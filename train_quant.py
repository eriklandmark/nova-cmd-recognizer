import tensorflow as tf

tf.enable_eager_execution()

import model
import data_generator
import os

BATCH_SIZE = 32
EPOCHS = 20
TOTAL_EXAMPLES = 14538
OUTPUT_PATH = "trained_models"

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
config.gpu_options.allow_growth = True
train_graph = tf.Graph()
train_sess = tf.compat.v1.Session(graph=train_graph, config=config)

tf.compat.v1.keras.backend.set_session(train_sess)

labels = data_generator.get_labels()

(x,y) = data_generator.generate_dataset(labels, use_images=True)


with train_graph.as_default():
    print("Loading model")
    labels = data_generator.get_labels()
    model = model.get_model(len(labels))

    tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=100)
    train_sess.run(tf.compat.v1.global_variables_initializer())

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                  metrics=['accuracy']
                  )

    model.summary()

    print("Starting Training")

    model.fit(x=x,y=y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, verbose=1, callbacks=[
            #tf.keras.callbacks.TensorBoard(log_dir="tensorboard", write_graph=False, update_freq=100),
            # tf.keras.callbacks.ModelCheckpoint(os.path.join(OUTPUT_PATH, "model.{epoch:02d}.ckpt"), save_weights_only=True,
            #                                   verbose=1)
        ])

    # model.save(os.path.join(OUTPUT_PATH, "model.h5"))
    saver = tf.train.Saver()
    saver.save(train_sess, 'quantized_model/checkpoints')
