import tensorflow as tf
import os
tf.enable_eager_execution()

import model
import data_generator

eval_graph = tf.Graph()
eval_sess = tf.Session(graph=eval_graph)

tf.keras.backend.set_session(eval_sess)

with eval_graph.as_default():
    tf.keras.backend.set_learning_phase(0)
    eval_model = model.get_model(len(data_generator.get_labels()))
    tf.contrib.quantize.create_eval_graph(input_graph=eval_graph)
    eval_graph_def = eval_graph.as_graph_def()
    saver = tf.train.Saver()
    saver.restore(eval_sess, 'quantized_model/checkpoints')

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        eval_sess,
        eval_graph_def,
        [eval_model.output.op.name]
    )

    with open('quantized_model/frozen_model.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    print(os.system("tflite_convert  --output_file=quantized_model/model.tflite --graph_def_file=quantized_model/frozen_model.pb --inference_type=QUANTIZED_UINT8 --input_arrays=input_1 --output_arrays=dense_1/Softmax --mean_values=0 --std_dev_values=255"))