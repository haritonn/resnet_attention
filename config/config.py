import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories
TRAIN_DIR = "./data/train"
TEST_DIR = "./data/test"
CHECKPOINTS_DIR = "./checkpoints"  # Checkpoints of trained model
TRAIN_RESULTS_DIR = "./train_results"  # Plots of train process

# Core parameters
EARLY_STOPPING_PATIENCE = 7
USING_AUGMENTATIONS = True
ATTENTION_REDUCTION = 16  # For channel attention mechanism

# Train parameters
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
CRITERION = (
    "CrossEntropyLoss"  # Using getattr to parse, so write like name in PyTorch module
)

# Other
RANDOM_SEED = 42
WEIGHT_DECAY = 2e-4  # L2
VAL_SPLIT = 0.2
NUM_CLASSES = 100  # CIFAR-100
SGD_MOMENTUM = 0.9
