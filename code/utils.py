import os
import psutil
import GPUtil
import shutil
import tempfile
import numpy as np
from typing import Dict, Tuple, Set, List, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import mxnet as mx
from gluonts.dataset.field_names import FieldName
from gluonts.mx import Trainer
from gluonts.core.component import validated
from gluonts.mx.trainer.callback import Callback

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

class EarlyStoppingCallback(Callback):
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float('inf')
        self.wait_count = 0
        self.best_params_path = None
        self.temp_dir = None
        
    def __del__(self):
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def on_validation_epoch_end(
        self,
        epoch_no: int,
        epoch_loss: float,
        training_network: mx.gluon.HybridBlock,
        trainer: mx.gluon.Trainer,
    ) -> bool:
        if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.wait_count = 0
            if self.temp_dir is None:
                self.temp_dir = tempfile.mkdtemp(prefix='early_stopping_')

            self.best_params_path = os.path.join(self.temp_dir, 'best_model.params')
            training_network.save_parameters(self.best_params_path)
        else:
            self.wait_count += 1
        if self.wait_count >= self.patience:
            print(f"Early stopping triggered")
            if self.best_params_path is not None:
                training_network.load_parameters(self.best_params_path)
                print("Restored best estimator")
            return False
        return True

class EarlyStoppingTrainer(Trainer):
    @validated()
    def __init__(self, patience: int, **kwargs):
        callbacks = kwargs.get('callbacks', [])
        callbacks.append(EarlyStoppingCallback(patience=patience))
        kwargs['callbacks'] = callbacks
        super().__init__(**kwargs)

class Normalizer:
    def __init__(self, fields: Set[str], method: str):
        self.fields = list(fields)
        self.stats: Dict[str, Dict[str, float]] = {}
        self.feature_indices: Dict[str, int] = {}
        self.method = method
        
    def _compute_stats(self, values: np.ndarray) -> Dict[str, float]:
        if self.method == "standard":
            center = float(np.mean(values))
            scale = float(np.std(values))
            if scale == 0:
                scale = 1.0
        elif self.method == "minmax":
            center = float(np.min(values))
            scale = float(np.max(values) - np.min(values))
            if scale == 0:
                scale = 1.0
        elif self.method == "robust":
            center = float(np.median(values))
            scale = float(np.percentile(values, 75) - np.percentile(values, 25))
            if scale == 0:
                scale = float(np.std(values))
                if scale == 0:
                    scale = 1.0
        return {"center": center, "scale": scale}
        
    def fit(self, dataset: List) -> None:
        target_values = []
        for data in dataset:
            target_values.extend(data[FieldName.TARGET].reshape(-1))
        target_values = np.array(target_values)
        self.stats["target"] = self._compute_stats(target_values)
        
        feat_values = {field: [] for field in self.fields}
        for data in dataset:
            dynamic_features = data[FieldName.FEAT_DYNAMIC_REAL]
            for i, field in enumerate(self.fields):
                feat_values[field].extend(dynamic_features[i].reshape(-1))
                self.feature_indices[field] = i
        for field in self.fields:
            values = np.array(feat_values[field])
            self.stats[field] = self._compute_stats(values)
    
    def normalize_array(self, arr: np.ndarray, field: str) -> np.ndarray:
        stats = self.stats[field if field != "sales_sum" else "target"]
        return (arr - stats["center"]) / stats["scale"]
    
    def transform(self, dataset: List) -> List:
        normalized_data = []
        for data in dataset:
            new_data = data.copy()
            new_data[FieldName.TARGET] = self.normalize_array(data[FieldName.TARGET], "sales_sum")
            
            dynamic_features = data[FieldName.FEAT_DYNAMIC_REAL]
            normalized_features = dynamic_features.copy()
            
            for field in self.fields:
                idx = self.feature_indices[field]
                normalized_features[idx] = self.normalize_array(dynamic_features[idx], field)
            
            new_data[FieldName.FEAT_DYNAMIC_REAL] = normalized_features
            normalized_data.append(new_data)
        
        return normalized_data
    
    def inverse_transform_labels(self, labels: np.ndarray) -> np.ndarray:
        stats = self.stats["target"]
        return labels * stats["scale"] + stats["center"]
        
    def inverse_transform_forecast(self, forecast) -> Any:
        stats = self.stats["target"] 
        if hasattr(forecast, 'samples'): 
            forecast.samples = forecast.samples * stats["scale"] + stats["center"]
        elif hasattr(forecast, 'forecast_keys'): 
            for q in forecast.forecast_keys:
                quantile_forecast = forecast.quantile(q)
                forecast._forecast_dict[q] = quantile_forecast * stats["scale"] + stats["center"]
        else:
            raise ValueError(f"Unknown forecast type: {type(forecast)}")
        return forecast

class TFTNormalizer:
    def __init__(self, fields: Set[str], method: str):
        self.fields = fields
        self.stats: Dict[str, Dict[str, float]] = {}
        self.method = method
        
    def _compute_stats(self, values: np.ndarray) -> Dict[str, float]:
        if self.method == "standard":
            center = float(np.mean(values))
            scale = float(np.std(values))
            if scale == 0:
                scale = 1.0
        elif self.method == "minmax":
            center = float(np.min(values))
            scale = float(np.max(values) - np.min(values))
            if scale == 0:
                scale = 1.0
        elif self.method == "robust":
            center = float(np.median(values))
            scale = float(np.percentile(values, 75) - np.percentile(values, 25))
            if scale == 0:
                scale = float(np.std(values))
                if scale == 0:
                    scale = 1.0

        return {"center": center, "scale": scale}
        
    def fit(self, dataset: List) -> None:
        target_values = []
        for data in dataset:
            target_values.extend(data[FieldName.TARGET].reshape(-1))
        self.stats["target"] = self._compute_stats(np.array(target_values))
        
        for feature in self.fields:
            feat_values = []
            for data in dataset:
                feat_values.extend(data[feature].reshape(-1))
            self.stats[feature] = self._compute_stats(np.array(feat_values))
    
    def normalize_array(self, arr: np.ndarray, field: str) -> np.ndarray:
        stats = self.stats[field]
        return (arr - stats["center"]) / stats["scale"]
    
    def transform(self, dataset: List) -> List:
        normalized_data = []
        for data in dataset:
            new_data = data.copy()
            new_data[FieldName.TARGET] = self.normalize_array(
                data[FieldName.TARGET], "target"
            )
            
            for field in self.fields:
                if field in data:
                    new_data[field] = self.normalize_array(data[field], field)
        
            normalized_data.append(new_data)
        
        return normalized_data
    
    def inverse_transform_labels(self, labels: np.ndarray) -> np.ndarray:
        stats = self.stats["target"]
        return labels * stats["scale"] + stats["center"]
        
    def inverse_transform_forecast(self, forecast) -> Any:
        stats = self.stats["target"] 
        if hasattr(forecast, 'samples'): 
            forecast.samples = forecast.samples * stats["scale"] + stats["center"]
        elif hasattr(forecast, 'forecast_keys'): 
            for q in forecast.forecast_keys:
                quantile_forecast = forecast.quantile(q)
                forecast._forecast_dict[q] = quantile_forecast * stats["scale"] + stats["center"]
        else:
            raise ValueError(f"Unknown forecast type: {type(forecast)}")
        return forecast

def normalize_dataset(
    train_dataset: List,
    valid_datasset: List,
    test_dataset: List,
    fields: Set[str] = {
        'sales_sum',
        'sales_mean', 'sales_std', 'sales_max', 'sales_min', 'sales_diff_mean',
        'sales_lag1_mean', 'sales_lag7_mean', 'sales_lag28_mean',
        'sales_rolling7_mean', 'sales_rolling28_mean', 'sales_rolling7_diff_mean', 'sales_rolling28_diff_mean',
        'release_mean', 'out_of_stock_mean',
        'sell_price_mean', 'sell_price_std', 'sell_price_max', 'sell_price_min', 'sell_price_diff_mean',
        'sell_price_lag_mean', 'sell_price_rolling_mean', 'sell_price_rolling_diff_mean',
        'sell_price_in_store_mean'
    },
    method: str = 'minmax',
) -> Tuple[List, List, Normalizer]:
    normalizer = Normalizer(fields, method)
        
    normalizer.fit(train_dataset)
    normalized_train = normalizer.transform(train_dataset)
    normalized_valid = normalizer.transform(valid_datasset)
    normalized_test = normalizer.transform(test_dataset)
    
    return normalized_train, normalized_valid, normalized_test, normalizer

def calculate_metrics(y_true, y_pred):
    abs_error = np.sum(np.abs(y_true - y_pred))
    abs_target_sum = np.sum(np.abs(y_true))
    abs_target_mean = np.mean(np.abs(y_true))
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'abs_error': abs_error,
        'abs_target_sum': abs_target_sum,
        'abs_target_mean': abs_target_mean,
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'sMAPE': 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))),
        'NRMSE': rmse / abs_target_mean,
        'ND': abs_error / abs_target_sum,
        'mean_absolute_error': mean_absolute_error(y_true, y_pred)
    }
    
    return metrics