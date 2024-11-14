import pandas as pd
import numpy as np
import random
import time
import os
from Compress import compress_zstd, compress_lz4, compress_gzip
from Decompress import decompress_zstd, decompress_lz4, decompress_gzip
from transformers import pipeline  
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import re

warnings.filterwarnings("ignore")

###############################################
column_name = 'Text'
model_name = "Roberta-base"  
mask_token = "<mask>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.mask_token = mask_token
# nrows = [1000, 5000, 10000]
nrows = [5000]
# tops = [3, 5, 10]
tops = [10]
row = 0


file_path = '../data/train_40k_text.csv'
temp_csv_path = '../temp/temp_train_40k_text.csv'
decompress_file_path = '../decompress/temp_train_40k.csv'
################################################

def random_mask(text, mask_percentage=0.2):
    words = text.split()
    num_to_mask = max(1, int(len(words) * mask_percentage))
    indices_to_mask = random.sample(range(len(words)), num_to_mask)
    for i in indices_to_mask:
        words[i] = mask_token  
    return ' '.join(words)

def mask_top_tfidf_words(texts, top_n=10):
    if not texts or all(t.isspace() for t in texts):
        return texts

    # Preprocess texts: remove punctuation and normalize
    processed_texts = [re.sub(r'[^\w\s]', '', text) for text in texts]

    # Calculate TF-IDF for the whole column
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(processed_texts)

    # Get feature names and their TF-IDF scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1

    # Get indices of top N TF-IDF words
    top_indices = np.argsort(tfidf_scores)[-top_n:][::-1]
    top_words = {feature_names[i]: mask_token for i in top_indices}

    # Print the top words with their TF-IDF scores
    print(f"Top {top_n} TF-IDF Words:")
    for word in top_words.keys():
        print(word)

    # Mask the top words in each text
    masked_texts = []
    for text in texts:
        for word in top_words.keys():
            text = text.replace(word, mask_token)
        masked_texts.append(text)

    return masked_texts  

"""
mask words with top10 frequency
"""
def mask_top_frequent_words(texts, top_n=10):
    if not texts or all(t.isspace() for t in texts):
        return texts

    # Preprocess texts: remove punctuation and normalize
    processed_texts = [re.sub(r'[^\w\s]', '', text.lower()) for text in texts]

    # Count word frequencies
    word_freq = {}
    for text in processed_texts:
        for word in text.split():
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    # Get top N most frequent words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_words = {word: mask_token for word, _ in top_words}

    # Print the top words with their frequencies
    print("Top 10 Most Frequent Words:")
    for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]:
        print(f"{word}: {freq}")

    # Mask the top words in each text
    masked_texts = []
    for text in texts:
        lower_text = text.lower()
        for word in top_words.keys():
            lower_text = re.sub(r'\b' + re.escape(word) + r'\b', mask_token, lower_text)
        masked_texts.append(lower_text)

    return masked_texts

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

# def predict_mask(text, fill_mask_pipeline):
#     global row
#     row += 1
#     if row % 100 == 0:
#         print(f'Predicting {row} row.')
#     masked_words = text.split(mask_token)
#     predictions = []
    
#     for i in range(len(masked_words) - 1):
#         input_text = masked_words[i] + mask_token + masked_words[i + 1]
#         results = fill_mask_pipeline(input_text)
        
#         predicted_word = results[0]['token_str'] 
#         predictions.append(predicted_word)

#     # Reconstruct the original text with predictions
#     reconstructed_text = ''
#     for i in range(len(masked_words) - 1):
#         reconstructed_text += masked_words[i]
#         reconstructed_text += predictions[i] 
#     reconstructed_text += masked_words[-1]

#     return reconstructed_text

def predict_mask(text, fill_mask_pipeline):
    global row
    row += 1
    if row % 100 == 0:
        print(f'Predicting {row} row.')
    
    # Split the text by the mask token
    masked_words = text.split(mask_token)
    predictions = []

    # Iterate over the parts and predict
    for i in range(len(masked_words) - 1):
        # Construct the input text with the mask in between
        input_text = masked_words[i] + mask_token + masked_words[i + 1]

        # Tokenize and truncate the input to fit model's max length
        input_tokens = tokenizer.tokenize(input_text)
        if len(input_tokens) > 512:
            input_tokens = input_tokens[:512]  # Truncate to 512 tokens
        truncated_input = tokenizer.convert_tokens_to_string(input_tokens)

        # Use the fill-mask pipeline to get predictions
        results = fill_mask_pipeline(truncated_input)

        # Extract the predicted word
        predicted_word = results[0]['token_str'] if results else ""
        predictions.append(predicted_word)

    # Reconstruct the original text with predictions
    reconstructed_text = ''
    for i in range(len(masked_words) - 1):
        reconstructed_text += masked_words[i]
        reconstructed_text += predictions[i] 
    reconstructed_text += masked_words[-1]  # Append the last part

    return reconstructed_text

# Function to count masks and differences
def count_masks_and_differences(original_text, masked_text, predicted_text):
    original_words = original_text.split()
    masked_words = masked_text.split()
    predicted_words = predicted_text.split()
    
    # Count masks
    masks_count = masked_words.count(mask_token)
    
    # Count differences only for masked positions
    differences_count = 0
    for i, (original, masked, predicted) in enumerate(zip(original_words, masked_words, predicted_words)):
        if masked == mask_token:  # Only count differences for masked tokens
            if original != predicted:
                differences_count += 1
    
    return masks_count, differences_count

def main():
    for nrow in nrows:
        for top in tops:

            ###########################################
            output_file_path = f'../Output/mask_freq_top{top}_{nrow}rows.csv'
            results_file_path = f'../Output/results_freq_top{top}_{nrow}rows.csv'  
            ###########################################

            data = pd.read_csv(file_path, nrows=nrow)
            original = pd.read_csv('../data/train_40k_text.csv')
            original_texts = original[column_name].copy()
            # print('original text: ', original_texts.head(10))
            data = data[data[column_name].str.split().str.len() > 10]

            # Mask the text in the DataFrame using the top 10 frequency words for the whole column
            data[column_name] = mask_top_frequent_words(data[column_name].tolist(), top_n=top)
            
            # Save the temporary CSV file
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
                global row
                row = 0
                compress_time, compression_ratio = compress_file(temp_csv_path, method=method)
                print(f"Temp file compressed using {method} in {compress_time:.2f} seconds with a compression ratio of {compression_ratio:.2f}.")

                decompressed_data = pd.read_csv(temp_csv_path)

                # Setup the mask prediction pipeline
                # print('mask token: ', tokenizer.mask_token)
                fill_mask_pipeline = pipeline("fill-mask", model=model_name, tokenizer=tokenizer, device=0)

                # Fill the masks in the Text column
                print(f"Mask predicting for {method}...")
                start = time.time()

                # Initialize counters
                total_masks = 0
                total_differences = 0

                # Apply prediction and count masks and differences
                decompressed_data["Predict_" + column_name] = decompressed_data[column_name].apply(lambda text: predict_mask(text, fill_mask_pipeline))

                # Calculate total masks and differences
                for original, masked, predicted in zip(original_texts, decompressed_data[column_name], decompressed_data["Predict_" + column_name]):
                    masks_count, differences_count = count_masks_and_differences(original, masked, predicted)
                    total_masks += masks_count
                    total_differences += differences_count

                end = time.time()
                
                elapsed_time = end - start
                print("Mask prediction completed in {:.2f} seconds.".format(elapsed_time))

                # Print total masks and differences
                print(f"Total masks: {total_masks}")
                print(f"Total different predicted words: {total_differences}")

                # Save the updated DataFrame with predictions
                decompressed_data.to_csv(output_file_path.replace('.csv', f'_{method}.csv'), index=False)  # Save output for each method

                # Display the updated DataFrame with comparisons
                print(decompressed_data[[column_name, "Predict_" + column_name]].head())
                
                # Store the compression results for the temporary CSV along with prediction time
                compression_results.append({
                    'method': method,
                    'compress_time': compress_time,
                    'compression_ratio': compression_ratio,
                    'prediction_time': elapsed_time,
                    'total masks': total_masks,
                    'total differences': total_differences
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