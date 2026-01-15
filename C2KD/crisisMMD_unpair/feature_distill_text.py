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
import ot

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

def compute_FA(stu_f, tea_f, device):
    batch_size = stu_f.size(0)
    a = torch.ones(batch_size, device=device) / batch_size
    b = torch.ones(batch_size, device=device) / batch_size

    eps = 1e-8
    stu_f_norm = stu_f / (stu_f.norm(dim=1, keepdim=True) + eps)
    tea_f_norm = tea_f / (tea_f.norm(dim=1, keepdim=True) + eps)

    M = 1.0 - torch.matmul(stu_f_norm, tea_f_norm.T)
    M = torch.clamp(M, min=0.0)

    fa = ot.sinkhorn2(
            a, b, M,
            reg=0.1,
            numItermax=100
        )
    return fa

def feature_distill_one_epoch(student_model, teacher_model, projector ,data_loader,  optimizer,  device, lambda_fa=1.0):
    student_model.train()
    teacher_model.eval()
    projector.train()
    criterion = nn.CrossEntropyLoss()
    
    fa_loss = 0.0
    ce_loss = 0.0
    total_loss = 0.0
    correct = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with torch.no_grad():
            _, tea_f = teacher_model(images, return_feature = True) 

        
        stu_logits, stu_f = student_model(input_ids, attention_mask, return_feature = True)
        ### adding projector
        stu_f = projector(stu_f)

        CE_loss = criterion(stu_logits, labels)
        FA_loss = compute_FA(stu_f, tea_f,device)
        
        loss = CE_loss + lambda_fa * FA_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_fa += CE_loss.item()
        total_ce += FA_loss.item()

        preds = torch.argmax(stu_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / len(data_loader),
        "ce": total_ce / len(data_loader),
        "fa": total_fa / len(data_loader),
        "acc": correct / total
    }


def train():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher_epochs = 1
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
                                              image_transform= image_transform,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    val_dataset = CrisisMMDDataset_unpaired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= image_transform,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    test_dataset = CrisisMMDDataset_unpaired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= image_transform,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    print("define the projector...")
    projector = FeatureProjector(
        in_dim= student_model.feature_dim,      # student hidden
        out_dim=teacher_model.feature_dim  # define in teacher
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(student_model.parameters()) + list(projector.parameters()),
        lr=3e-4
    )
    student_epochs = 10
    for epoch in range(student_epochs):
        feature_distill_one_epoch(student_model, teacher_model, projector, train_dataset, optimizer, device)


if __name__ == "__main__":
    train()