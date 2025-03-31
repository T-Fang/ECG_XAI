import sys
import platform

CURRENT_OS = platform.system()
<<<<<<< Updated upstream
PROJECT_PATH = '/home/ftian/storage/projects/ECG_XAI/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer_Science/Archive/FYP/ECG_XAI/'
=======
PROJECT_PATH = '/data2/haoyang/ECG_XAI/' if CURRENT_OS == 'Linux' else '/Users/tf/Computer_Science/Archive/FYP/ECG_XAI/'
>>>>>>> Stashed changes
sys.path.insert(1, PROJECT_PATH)

from src.utils.train_utils import seed_all  # noqa: E402

seed_all()
