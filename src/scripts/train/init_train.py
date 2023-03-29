import sys
import platform

CURRENT_OS = platform.system()
PROJECT_PATH = '/home/ftian/storage/ECG_XAI/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer_Science/Archive/FYP/ECG_XAI/'
sys.path.insert(1, PROJECT_PATH)

from src.utils.train_utils import seed_all, set_cuda_env  # noqa: E402

seed_all()
set_cuda_env(gpu_ids='1')
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# for i in range(torch.cuda.device_count()):
#     info = torch.cuda.get_device_properties(i)
#     print(f"CUDA:{i} {info.name}, {info.total_memory / 1024 ** 2}MB")
# # torch.cuda.set_device(0)
# torch.set_float32_matmul_precision('medium')
