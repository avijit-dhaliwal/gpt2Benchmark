import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from python.gpt2 import GPT2Advanced

def export_to_onnx(model, input_size, file_path):
    dummy_input = torch.randint(0, 50257, (1, input_size))
    torch.onnx.export(model, dummy_input, file_path, 
                      input_names=['input'], 
                      output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size', 1: 'sequence'},
                                    'output': {0: 'batch_size', 1: 'sequence'}})

if __name__ == "__main__":
    model = GPT2Advanced(50257, 768, 12, 12, 1024)
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)
    
    export_to_onnx(model, 100, 'models/gpt2_model.onnx')
    print("ONNX model exported successfully.")