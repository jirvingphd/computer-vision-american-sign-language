"""
Example usage:
import custom_functions.quantize_model as qm

model_path = 'path/to/your/model'
output_path = 'path/to/save/quantized_model.tflite'

# # Create a TensorFlow dataset for the representative data
# representative_dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([100, 224, 224, 3]))

# Convert the model to quantized TensorFlow Lite format
convert_to_quantized_tflite(model_path, output_path, representative_dataset)

# Load the quantized model
interpreter = load_quantized_model(output_path)

# Prepare the input data for inference
input_shape = interpreter.get_input_details()[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Run inference
output_data = run_inference(interpreter, input_data)
print(output_data)
"""
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import numpy as np

def convert_to_quantized_tflite(model_path, output_path, representative_dataset=None, return_model=False):
    """
    Convert a TensorFlow model to a quantized TensorFlow Lite model.
    
    Parameters:
    - model_path (str): Path to the saved TensorFlow model.
    - output_path (str): Path to save the quantized TensorFlow Lite model.
    - representative_dataset (tf.data.Dataset): Optional representative dataset for full integer quantization.
    """
    # Load the pre-trained model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model to TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    if representative_dataset:
        def representative_data_gen():
            for data,_ in representative_dataset.take(100):
                yield [data]

        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8  # Ensure correct input type
        converter.inference_output_type = tf.uint8  # Ensure correct output type
    tflite_model = converter.convert()
    
    # Save the quantized model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Quantized model saved to {output_path}")
    if return_model:
        return tflite_model

def load_quantized_model(model_path):
    """
    Load a quantized TensorFlow Lite model for inference.
    
    Parameters:
    - model_path (str): Path to the quantized TensorFlow Lite model.
    
    Returns:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
    """
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    
    # Allocate tensors
    interpreter.allocate_tensors()
    
    return interpreter

def run_inference(interpreter, input_data):
    """
    Run inference on a loaded TensorFlow Lite model.
    
    Parameters:
    - interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter.
    - input_data (np.array): The input data for inference.
    
    Returns:
    - output_data (np.array): The output data from the inference.
    """
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = np.array(input_data, dtype=np.uint8)  
    # Set the tensor to point to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run the inference
    interpreter.invoke()
    
    # Get the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

# # Example usage
# if __name__ == "__main__":
#     model_path = 'path/to/your/model'
#     output_path = 'path/to/save/quantized_model.tflite'
    
#     # Create a TensorFlow dataset for the representative data
#     representative_dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([100, 224, 224, 3]))

#     # Convert the model to quantized TensorFlow Lite format
#     convert_to_quantized_tflite(model_path, output_path, representative_dataset)

#     # Load the quantized model
#     interpreter = load_quantized_model(output_path)

#     # Prepare the input data for inference
#     input_shape = interpreter.get_input_details()[0]['shape']
#     input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

#     # Run inference
#     output_data = run_inference(interpreter, input_data)
#     print(output_data)
