import os
import shutil
import tempfile
import numpy as np
from typing import Dict, Tuple, Set, List, Any

import mxnet as mx
from gluonts.dataset.field_names import FieldName
from gluonts.mx import Trainer
from gluonts.core.component import validated
from gluonts.mx.trainer.callback import Callback

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
    normalized_test = normalizer.transform(test_dataset)
    
    return normalized_train, normalized_test, normalizer
