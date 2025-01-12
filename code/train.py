import torch
from dataset import dataset
from utils import *

# In[ ]:

model_name = 
comment = ''
trial = '1'

# In[ ]:

### set default config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_log_path, base_log_path, trial_path = get_log_path(f"m5_submission/{model_name}", comment, trial)

### set logger
set_logger(full_log_path)

# In[ ]:

### make dataset
train_ds, val_ds, stat_cat_cardinalities = dataset(data_path='../data/', exclude_no_sales=True)

# In[ ]:

### get estimator
logger = logging.getLogger("mofl").getChild("training")
logger.info(f"Using {model_name} model...")
estimator = get_estimator(model_name, stat_cat_cardinalities, device, base_log_path, full_log_path)

### path for trained model
model_path = Path(full_log_path+"/trained_model")
model_path.mkdir()

# In[ ]:

### prediction
predictor = estimator.train(train_ds, validation_period=10)

# In[ ]:

### save model
logger.info(f"Save {model_name} model...")
model_path = Path(full_log_path+"/predictor")        
model_path.mkdir()
predictor.serialize(model_path)