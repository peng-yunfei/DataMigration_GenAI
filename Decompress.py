'''
Functions for data decompression
Return compression time, original data size, and decompressed data size
'''
import gzip
import shutil
import lz4.frame
import zstandard as zstd
import os
import time

output_dir = '../decompress/'
os.makedirs(output_dir, exist_ok=True)

def remove_existing_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)

def decompress_zstd(file_path):
    output_zstd = os.path.join(output_dir, os.path.basename(file_path).replace('.zst', '')) 

    remove_existing_file(output_zstd)

    dctx = zstd.ZstdDecompressor()

    start_time = time.time()
    with open(file_path, 'rb') as f_in:
        with open(output_zstd, 'wb') as f_out:
            dctx.copy_stream(f_in, f_out)
    end_time = time.time()

    original_size = os.path.getsize(file_path)
    decompressed_size = os.path.getsize(output_zstd)

    return end_time - start_time, original_size, decompressed_size

def decompress_lz4(file_path):
    output_lz4 = os.path.join(output_dir, os.path.basename(file_path).replace('.lz4', ''))  # Remove .lz4 extension

    remove_existing_file(output_lz4)

    start_time = time.time()
    with open(file_path, 'rb') as f_in:
        with lz4.frame.open(output_lz4, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    end_time = time.time()

    original_size = os.path.getsize(file_path)
    decompressed_size = os.path.getsize(output_lz4)

    return end_time - start_time, original_size, decompressed_size

def decompress_gzip(file_path):
    output_gzip = os.path.join(output_dir, os.path.basename(file_path).replace('.gz', ''))  # Remove .gz extension

    remove_existing_file(output_gzip)

    start_time = time.time()
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_gzip, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    end_time = time.time()

    original_size = os.path.getsize(file_path)
    decompressed_size = os.path.getsize(output_gzip)

    return end_time - start_time, original_size, decompressed_size
