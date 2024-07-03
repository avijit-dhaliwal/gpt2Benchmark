import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def benchmark_python(prompt, max_length):
    url = 'http://localhost:8000/generate'
    start_time = time.time()
    try:
        response = requests.post(url, json={"prompt": prompt, "max_length": max_length})
        response.raise_for_status()
        end_time = time.time()
        response_json = response.json()
        if 'generated_text' not in response_json:
            print(f"Warning: 'generated_text' not in response. Full response: {response_json}")
            return end_time - start_time, str(response_json)
        return end_time - start_time, response_json['generated_text']
    except requests.RequestException as e:
        print(f"Error with request: {e}")
        return None, str(e)

def run_python_benchmarks(prompts, max_lengths, num_runs=3):
    results = []
    generated_texts = []

    for prompt in prompts:
        for max_length in max_lengths:
            print(f"Prompt: {prompt}, Max Length: {max_length}")
            run_times = []
            for run in range(num_runs):
                time_taken, text = benchmark_python(prompt, max_length)
                if time_taken is not None:
                    run_times.append(time_taken)
                    if run == 0:  # Only save text from first run
                        generated_texts.append(text)
                    print(f"Run {run + 1} Time: {time_taken:.4f}s")
                else:
                    print(f"Run {run + 1} failed: {text}")
            if run_times:
                avg_time = np.mean(run_times)
                results.append({'prompt': prompt, 'max_length': max_length, 'avg_time': avg_time})
                print(f"Average Time: {avg_time:.4f}s")
            print("--------------------")

    return results, generated_texts

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

if __name__ == "__main__":
    prompts = ["Hello, world!", "The quick brown fox", "In a galaxy far, far away"]
    max_lengths = [50, 100, 200]

    results, generated_texts = run_python_benchmarks(prompts, max_lengths)
    np.save('python_benchmark_results.npy', {
        'results': results,
        'generated_texts': generated_texts,
        'prompts': prompts,
        'max_lengths': max_lengths
    })

    visualize_results(results)

    
    print("Benchmark complete. Results saved and visualized.")