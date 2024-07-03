import onnxruntime as ort
import numpy as np

# Load the ONNX model
ort_session = ort.InferenceSession("models/gpt2_model.onnx")

# Create a random input
input_data = np.random.randint(0, 50257, (1, 100)).astype(np.int64)

# Run inference
ort_inputs = {ort_session.get_inputs()[0].name: input_data}
ort_outputs = ort_session.run(None, ort_inputs)

print("ONNX model output shape:", ort_outputs[0].shape)