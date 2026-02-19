import torch
from sklearn.metrics import f1_score

def compute_f1(preds, labels):
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).int().cpu().numpy()
    labels = labels.cpu().numpy()
    
    micro = f1_score(labels, preds, average="micro")
    macro = f1_score(labels, preds, average="macro")
    
    return micro, macro
