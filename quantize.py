import tensorflow as tf
import data_generator

model = tf.keras.models.load_model("trained_models_v2/model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = "QUANTIZED_UINT8"
converter.inference_output_type = "QUANTIZED_UINT8"

tflite_model = converter.convert()
tflite_model_file = "quantized_model/model.tflite"
with open(tflite_model_file, "wb") as tf_file:
    tf_file.write(tflite_model)