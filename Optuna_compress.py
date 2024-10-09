import optuna
import zstandard as zstd
import lz4.frame
import gzip
import time
import os

file_path = "../data/DS_001.csv"


def compress_lz4(file_path, output_path, compression_level, block_size):
    with open(file_path, 'rb') as f_in:
        with lz4.frame.open(output_path, mode='wb', compression_level=compression_level, block_size=block_size * 1024) as f_out:
            f_out.write(f_in.read())  


def compression_objective(trial):
    compression_algorithm = trial.suggest_categorical('algorithm', ['zstd', 'lz4', 'gzip'])
    compression_level = trial.suggest_int('compression_level', 1, 9)  # Compression levels
    block_size = trial.suggest_int('block_size', 128, 1024)  # Block size in KB (64KB to 1024KB)
    # dictionary_size = trial.suggest_int('dictionary_size', 1024, 4096)  # Dictionary size (1MB to 4MB)

    start_time = time.time()
    
    output_path = "../compressed_data/compressed_output"
    if compression_algorithm == 'zstd':
        cctx = zstd.ZstdCompressor(level=compression_level)
        with open(file_path, 'rb') as f_in:
            data = f_in.read()
            compressed_data = cctx.compress(data)
        compressed_size = len(compressed_data)

    elif compression_algorithm == 'lz4':
        output_path += '.lz4'
        compress_lz4(file_path, output_path, compression_level, block_size)
        compressed_size = os.path.getsize(output_path)
        
    elif compression_algorithm == 'gzip':
        output_path += '.gz'
        with open(file_path, 'rb') as f_in:
            with gzip.open(output_path, 'wb', compresslevel=compression_level) as f_out:
                f_out.write(f_in.read())
        compressed_size = os.path.getsize(output_path)

    time_taken = time.time() - start_time

    return time_taken * compressed_size


study = optuna.create_study(direction='minimize')
study.optimize(compression_objective, n_trials=5)

print(f"Best trial: {study.best_trial}")
print(f"Best params: {study.best_trial.params}")
