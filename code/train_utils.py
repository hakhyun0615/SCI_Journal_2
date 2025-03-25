import os
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