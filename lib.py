import tensorflow as tf

def progress_bar(iteration, total, prefix='', suffix='Completed', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '=' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    if iteration == total:
        print()

tf.keras.backend.set_learning_phase(0)

def freeze_graph():
    graph = tf.Graph()
    session = tf.compat.v1.Session(graph=graph)
    tf.compat.v1.keras.backend.set_session(session)

    with graph.as_default():
        model = tf.keras.models.load_model("trained_models_v2/model.h5")

        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()))
        output_names = [out.op.name for out in model.outputs]
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        for node in input_graph_def.node:
            node.device = ""
        frozen_graph = tf.compat.v1.graph_util.extract_sub_graph(input_graph_def, output_names)
        tf.io.write_graph(frozen_graph, "quantized_model", "saved_model.pbtxt", as_text=True)