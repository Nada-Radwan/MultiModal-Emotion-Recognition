import torch

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TEXT_MODEL_NAME = "bert-base-uncased"
    MAX_LENGTH = 128
    
    NUM_LABELS = 28  # GoEmotions
    
    BATCH_SIZE = 4
    LR = 2e-5
    EPOCHS = 1
    
    IMAGE_SIZE = 224
