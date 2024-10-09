import zstandard as zstd
import os
import time

def compress_zstd_best_params(file_path, output_path, compression_level, block_size):
    original_size = os.path.getsize(file_path)
    start_time = time.time()

    with open(file_path, 'rb') as f_in:
        data = f_in.read()

    cctx = zstd.ZstdCompressor(level=compression_level)
    compressed_data = cctx.compress(data)
    
    with open(output_path, 'wb') as f_out:
        f_out.write(compressed_data)

    end_time = time.time()
    
    compressed_size = os.path.getsize(output_path)
    compression_time = end_time - start_time
    compression_ratio = original_size / compressed_size

    print(f"Original file size: {original_size} bytes")
    print(f"Compressed file size: {compressed_size} bytes")
    print(f"Compression time: {compression_time:.4f} seconds")
    print(f"Compression ratio: {compression_ratio:.2f}")

input_file_path = "../data/DS_001.csv"
output_file_path = "../compressed_data/DS_001.zst"
best_compression_level = 4
best_block_size = 334

compress_zstd_best_params(input_file_path, output_file_path, best_compression_level, best_block_size)
