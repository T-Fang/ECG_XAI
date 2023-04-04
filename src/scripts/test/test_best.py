import os
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import init_test  # noqa: F401
from src.models.ecg_step_module import EcgPipeline
from src.utils.data_utils import EcgDataModule
from src.utils.train_utils import get_common_trainer_params, set_cuda_env
from src.basic.constants import BEST_MODEL_PATH

save_dir = BEST_MODEL_PATH
ckpt_name = 'MPAV_rho8-auroc=0.8360216.ckpt'
ckpt_path = os.path.join(save_dir, ckpt_name)

set_cuda_env(gpu_ids='0')
model = EcgPipeline.load_from_checkpoint(ckpt_path)
trainer = Trainer(logger=TensorBoardLogger(save_dir=save_dir), **get_common_trainer_params())
ecg_datamodule = EcgDataModule(batch_size=512, num_workers=0)
trainer.test(model, datamodule=ecg_datamodule)
