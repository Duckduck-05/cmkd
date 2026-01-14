from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from utils.MMDDataset import CrisisMMDTextDataset, CrisisMMDHumanitarianTextDataset, CrisisMMDDataset_unpaired
import torch.nn as nn
from torch.optim import AdamW
import torch 
from model.Bert_based import SmallBertStudent, FeatureProjector
from transformers import BertForSequenceClassification
from transformers import AutoModelForSequenceClassification
from train_mobilenet_teacher import get_pretraining_techer_model
from torchvision import transforms
from torch.utils.data import DataLoader


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

HUMANITARIAN_LABEL2ID = {
    "affected_individuals": 0,
    "infrastructure_and_utility_damage": 1,
    "injured_or_dead_people": 2,
    "missing_or_found_people": 3,
    "rescue_volunteering_or_donation_effort": 4,
    "vehicle_damage": 5,
    "other_relevant_information": 6,
    "not_humanitarian": 7,
}
HUMANITARIAN_ID2LABEL = {v: k for k, v in HUMANITARIAN_LABEL2ID.items()}


def feature_distill_one_epoch(student_model, teacher_model, projector ,data_loader,  optimizer,  device, lambda_fa=1.0):
    student_model.train()
    teacher_model.eval()
    projector.train()
    criterion = nn.CrossEntropyLoss()
    
    fa_loss = 0.0
    ce_loss = 0.0
    total_loss = 0.0
    correct = 0
    

def train():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher_epochs = 10
    print("pre-training vision teacher with epochs: ",teacher_epochs)
    teacher_model = get_pretraining_techer_model(epochs= teacher_epochs)
    print("frozen teacher model...")
    teacher_model.eval()
    print("Init student model...")
    device = "cuda"
    student_model = SmallBertStudent().to(device)
    print("Load unpaired data...")
    train_dataset = CrisisMMDDataset_unpaired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= transforms,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    val_dataset = CrisisMMDDataset_unpaired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= transforms,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    test_dataset = CrisisMMDDataset_unpaired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= transforms,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    print("define the projector...")
    projector = FeatureProjector(
        in_dim= student_model.feature_dim,      # student hidden
        out_dim=teacher_model.feature_dim  # define in teacher
    ).to(device)



if __name__ == "__main__":
    train()