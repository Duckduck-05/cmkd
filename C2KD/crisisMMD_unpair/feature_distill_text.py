from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from utils.MMDDataset import CrisisMMDTextDataset, CrisisMMDHumanitarianTextDataset
import torch.nn as nn
from torch.optim import AdamW
import torch 
from model.Bert_based import BertClassifier, BertModel
from transformers import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from train_mobilenet_teacher import get_pretraining_techer_model
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train():
    teacher_epochs = 10
    print("pre-training vision teacher with epochs: ",teacher_epochs)
    teacher_model = get_pretraining_techer_model(epochs= teacher_epochs)
    print("frozen teacher model...")
    teacher_model.eval()
    