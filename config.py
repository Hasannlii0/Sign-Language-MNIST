from pathlib import Path 
import torch


BASE_DIR = Path(__file__).resolve().parent 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4


IMAGE_SIZE = 224
NUM_CLASSES = 24


########################################
# Label mapping
########################################

LABELS = {
    0:'A',
    1:'B',
    2:'C',
    3:'D',
    4:'E',
    5:'F',
    6:'G',
    7:'H',
    8:'I',
    9:'K',
    10:'L',
    11:'M',
    12:'N',
    13:'O',
    14:'P',
    15:'Q',
    16:'R',
    17:'S',
    18:'T',
    19:'U',
    20:'V',
    21:'W',
    22:'X',
    23:'Y'
}

TRAIN_CSV = BASE_DIR / 'data' / 'sign_mnist_train.csv'
TEST_CSV = BASE_DIR / 'data' / 'sign_mnist_test.csv'

SAVE_PATH = BASE_DIR / 'checkpoints' / 'best_model.pth'