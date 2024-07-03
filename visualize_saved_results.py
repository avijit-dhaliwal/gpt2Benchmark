import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(results):
    df = pd.DataFrame(results)
    
    # Convert max_length to string
    df['max_length'] = df['max_length'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='prompt', y='avg_time', hue='max_length', data=df)
    plt.title('Python GPT-2 Generation Time')
    plt.xlabel('Prompt')
    plt.ylabel('Average Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Max Length')
    plt.tight_layout()
    plt.savefig('python_gpt2_benchmark.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='max_length', y='avg_time', hue='prompt', data=df.astype({'max_length': int}), marker='o')
    plt.title('Python GPT-2 Generation Time vs Max Length')
    plt.xlabel('Max Length')
    plt.ylabel('Average Time (seconds)')
    plt.legend(title='Prompt', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('python_gpt2_benchmark_line.png')
    plt.close()

# Load the saved results
loaded_data = np.load('python_benchmark_results.npy', allow_pickle=True).item()

results = loaded_data['results']
generated_texts = loaded_data['generated_texts']
prompts = loaded_data['prompts']
max_lengths = loaded_data['max_lengths']

# Visualize the results
visualize_results(results)

print("Visualizations created from saved results.")

# If you want to see the results:
for res in results:
    print(f"Prompt: {res['prompt']}, Max Length: {res['max_length']}, Avg Time: {res['avg_time']:.4f}s")