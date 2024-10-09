import pandas as pd
import time
import warnings
from transformers import GPT2Tokenizer
from Compress import compress_zstd, compress_lz4, compress_gzip

warnings.filterwarnings("ignore")

file_path = "../data/DS_001.csv"
output_path = "../data/tokenized_DS_001.csv"
loop_time = 5
text_column = "Content"

def tokenize(file_path, text_column):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    df = pd.read_csv(file_path, sep='|')

    start_time = time.time()
    df[text_column] = df[text_column].apply(lambda x: tokenizer.encode(x, return_tensors="pt").tolist())
    elapsed_time = time.time() - start_time

    # print(df[text_column].head(5))
    print(f"Time taken to tokenize: {elapsed_time:.4f} seconds")
    df.to_csv(output_path, index=False)

# Original Compression
def compress(file_path, text_column):
    z_time, z_cr, l_time, l_cr, g_time, g_cr = [], [], [], [], [], []
    for i in range(loop_time):
        print(f"Compressing {i} time...")
        ztime, zcr = compress_zstd(file_path)
        ltime, lcr = compress_lz4(file_path)
        gtime, gcr = compress_gzip(file_path)
        z_time.append(ztime)
        z_cr.append(zcr)
        l_time.append(ltime)
        l_cr.append(lcr)
        g_time.append(gtime)
        g_cr.append(gcr)

    # Calculate averages
    average_z_time = sum(z_time) / loop_time
    average_z_cr = sum(z_cr) / loop_time
    average_l_time = sum(l_time) / loop_time
    average_l_cr = sum(l_cr) / loop_time
    average_g_time = sum(g_time) / loop_time
    average_g_cr = sum(g_cr) / loop_time

    # Print the results
    print(file_path)
    print(f"Average Zstandard Time: {average_z_time:.4f} seconds")
    print(f"Average Zstandard Compression Ratio: {average_z_cr:.2f}")
    print(f"Average LZ4 Time: {average_l_time:.4f} seconds")
    print(f"Average LZ4 Compression Ratio: {average_l_cr:.2f}")
    print(f"Average Gzip Time: {average_g_time:.4f} seconds")
    print(f"Average Gzip Compression Ratio: {average_g_cr:.2f}")


def main():
    # tokenize(file_path, text_column)
    compress(file_path, text_column)
    compress(output_path, text_column)

if __name__ == "__main__":
    main()