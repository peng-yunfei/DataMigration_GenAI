import gzip
import shutil
import lz4.frame
import zstandard as zstd
import os
import time

# Create the output directory if it doesn't exist
output_dir = '../compressed_data/'
os.makedirs(output_dir, exist_ok=True)

def remove_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        # print(f"Removed existing file: {file_path}")

def compress_zstd(file_path):
    output_zstd = os.path.join(output_dir, os.path.basename(file_path) + '.zst')
    
    remove_existing_file(output_zstd)

    cctx = zstd.ZstdCompressor(level=3)  

    start_time = time.time()  
    with open(file_path, 'rb') as f_in:
        with open(output_zstd, 'wb') as f_out:
            f_out.write(cctx.compress(f_in.read()))
    end_time = time.time()  

    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(output_zstd)
    compression_ratio = original_size / compressed_size

    # print(f"Zstandard compressed file saved to {output_zstd}")
    # print(f"Time taken: {end_time - start_time:.4f} seconds")
    # print(f"Compression ratio: {compression_ratio:.2f}\n")

    return end_time - start_time, compression_ratio

def compress_lz4(file_path):
    output_lz4 = os.path.join(output_dir, os.path.basename(file_path) + '.lz4')

    remove_existing_file(output_lz4)

    start_time = time.time()  
    with open(file_path, 'rb') as f_in:
        with lz4.frame.open(output_lz4, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    end_time = time.time()  

    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(output_lz4)
    compression_ratio = original_size / compressed_size

    # print(f"LZ4 compressed file saved to {output_lz4}")
    # print(f"Time taken: {end_time - start_time:.4f} seconds")
    # print(f"Compression ratio: {compression_ratio:.2f}\n")

    return end_time - start_time, compression_ratio

def compress_gzip(file_path):
    output_gzip = os.path.join(output_dir, os.path.basename(file_path) + '.gz')

    remove_existing_file(output_gzip)

    start_time = time.time()  
    with open(file_path, 'rb') as f_in:
        with gzip.open(output_gzip, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    end_time = time.time()  

    original_size = os.path.getsize(file_path)
    compressed_size = os.path.getsize(output_gzip)
    compression_ratio = original_size / compressed_size

    # print(f"Gzip compressed file saved to {output_gzip}")
    # print(f"Time taken: {end_time - start_time:.4f} seconds")
    # print(f"Compression ratio: {compression_ratio:.2f}\n")

    return end_time - start_time, compression_ratio