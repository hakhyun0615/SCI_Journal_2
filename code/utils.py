import psutil
import GPUtil
import numpy as np

def highlight_print(text, color='yellow'):
    colors = {
        'yellow': '\033[93m',
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m'
    }
    color_code = colors.get(color, '\033[93m')
    print(color_code + str(text) + '\033[0m')

def reduce_memory(df, verbose=False):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def get_optimal_num_batches(gpu_available, verbose=False):
    # memory
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total
    available_memory = memory_info.available

    # cpu
    cpu_usage = psutil.cpu_percent(interval=None)

    # gpu
    if gpu_available:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_memory = sum(gpu.memoryTotal for gpu in gpus)
                gpu_memory_available = sum(gpu.memoryFree for gpu in gpus)
            else:
                gpu_memory = 0
                gpu_memory_available = 0
        except ImportError:
            gpu_memory = 0
            gpu_memory_available = 0
    else:
        gpu_memory = 0
        gpu_memory_available = 0

    # optimal
    memory_factor = available_memory / total_memory
    cpu_factor = (100 - cpu_usage) / 100
    gpu_factor = gpu_memory_available / max(gpu_memory, 1) if gpu_available else 1
    optimal_num_batches = int(200 * (0.5 * memory_factor + 0.3 * cpu_factor + 0.2 * gpu_factor))

    if verbose: print(f"Optimal num batches: {optimal_num_batches}")
    
    return max(1, optimal_num_batches)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"