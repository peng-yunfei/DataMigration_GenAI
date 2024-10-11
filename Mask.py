import pandas as pd
import numpy as np
import random
import time
import os
from Compress import compress_zstd, compress_lz4, compress_gzip
from Decompress import decompress_zstd, decompress_lz4, decompress_gzip
from transformers import pipeline  
from transformers import AutoTokenizer
import warnings

warnings.filterwarnings("ignore")

###############################################
file_path = '../data/train_40k.csv'
temp_csv_path = '../temp/temp_train_40k.csv'
decompress_file_path = '../decompress/temp_train_40k.csv'
output_file_path = '../Output/mask.csv'
results_file_path = '../Output/compression_results.csv'  # File to save results

column_name = 'Text'
model_name = "bert-base-uncased"  
mask_token = "[MASK]"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.mask_token = mask_token
row = 0
################################################

def random_mask(text, mask_percentage=0.2):
    words = text.split()
    num_to_mask = max(1, int(len(words) * mask_percentage))
    indices_to_mask = random.sample(range(len(words)), num_to_mask)
    for i in indices_to_mask:
        words[i] = mask_token  
    return ' '.join(words)

def compress_file(file_path, method='gzip'):
    if method == 'gzip':
        time, cr = compress_gzip(file_path)
    elif method == 'lz4':
        time, cr = compress_lz4(file_path)
    elif method == 'zstd':
        time, cr = compress_zstd(file_path)

    return time, cr

def decompress_file(file_path, method='gzip'):
    if method == 'gzip':
        decompress_gzip(file_path)
    elif method == 'lz4':
        decompress_lz4(file_path)
    elif method == 'zstd':
        decompress_zstd(file_path)

def predict_mask(text, fill_mask_pipeline):
    global row
    row += 1
    if row % 100 == 0:
        print(f'Predicting {row} row.')
    masked_words = text.split(mask_token)
    predictions = []
    
    for i in range(len(masked_words) - 1):
        input_text = masked_words[i] + mask_token + masked_words[i + 1]
        results = fill_mask_pipeline(input_text)
        
        predicted_word = results[0]['token_str'] 
        predictions.append(predicted_word)

    # Reconstruct the original text with predictions
    reconstructed_text = ''
    for i in range(len(masked_words) - 1):
        reconstructed_text += masked_words[i]
        reconstructed_text += predictions[i] 
    reconstructed_text += masked_words[-1]

    return reconstructed_text

def main():
    # data = pd.read_csv(file_path, nrows=20000)
    data = pd.read_csv(file_path)
    # Mask the text in the DataFrame
    data[column_name] = data[column_name].apply(random_mask)
    data.to_csv(temp_csv_path, index=False)
    print("Temp CSV file created successfully.")

    # List of compression methods
    compression_methods = ['gzip', 'lz4', 'zstd']
    
    # Store results for the original file
    original_compression_results = []

    # Compress the original file
    for method in compression_methods:
        original_compress_time, original_compression_ratio = compress_file(file_path, method=method)
        print(f"Original file compressed using {method} in {original_compress_time:.2f} seconds with a compression ratio of {original_compression_ratio:.2f}.")
        
        # Store the results for the original file
        original_compression_results.append({
            'method': method,
            'compress_time': original_compress_time,
            'compression_ratio': original_compression_ratio
        })

    # Store results for the temporary CSV file
    compression_results = []

    # Proceed with compressing the temporary CSV file
    for method in compression_methods:
        # Compress the temporary CSV file
        global row
        row = 0
        compress_time, compression_ratio = compress_file(temp_csv_path, method=method)
        print(f"Temp file compressed using {method} in {compress_time:.2f} seconds with a compression ratio of {compression_ratio:.2f}.")

        # Decompress the compressed file
        # decompress_file(temp_csv_path + f'.{method}', method=method) 
        # print(f"Temp file decompressed using {method}.")

        decompressed_data = pd.read_csv(temp_csv_path)

        # Setup the mask prediction pipeline
        fill_mask_pipeline = pipeline("fill-mask", model=model_name, tokenizer=tokenizer, device=0)

        # Fill the masks in the Text column
        print(f"Mask predicting for {method}...")
        start = time.time()
        decompressed_data["Predict_" + column_name] = decompressed_data[column_name].apply(lambda text: predict_mask(text, fill_mask_pipeline))
        end = time.time()
        
        elapsed_time = end - start
        print("Mask prediction completed in {:.2f} seconds.".format(elapsed_time))

        # Compare the original and predicted values
        decompressed_data['Difference'] = decompressed_data[column_name] != decompressed_data["Predict_" + column_name]
        decompressed_data.to_csv(output_file_path.replace('.csv', f'_{method}.csv'))  # Save output for each method

        # Display the updated DataFrame with comparisons
        print(decompressed_data[[column_name, "Predict_" + column_name, 'Difference']].head())
        
        # Count the number of True and False in the Difference column
        difference_counts = decompressed_data['Difference'].value_counts()
        print(f"Difference counts for {method}:\n{difference_counts}\n")

        # Store the compression results for the temporary CSV along with prediction time
        compression_results.append({
            'method': method,
            'compress_time': compress_time,
            'compression_ratio': compression_ratio,
            'prediction_time': elapsed_time  # Record prediction time
        })
    
    # Summary of compression results for the original file
    print("\nOriginal File Compression Results:")
    for result in original_compression_results:
        print(f"Method: {result['method']}, Compress Time: {result['compress_time']:.2f} seconds, Compression Ratio: {result['compression_ratio']:.2f}")

    # Summary of compression results for the temporary CSV
    print("\nTemporary CSV Compression Results:")
    for result in compression_results:
        print(f"Method: {result['method']}, Compress Time: {result['compress_time']:.2f} seconds, Compression Ratio: {result['compression_ratio']:.2f}, Prediction Time: {result['prediction_time']:.2f} seconds")

    # Save all results to a CSV file
    all_results = []

    for result in original_compression_results:
        all_results.append({**result, 'file_type': 'original'})

    for result in compression_results:
        all_results.append({**result, 'file_type': 'temporary'})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_file_path, index=False)
    print(f"\nAll results saved to {results_file_path}")

if __name__ == "__main__":
    main()
