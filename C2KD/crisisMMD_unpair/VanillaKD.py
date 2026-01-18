from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from utils.MMDDataset import CrisisMMDTextDataset, CrisisMMDHumanitarianTextDataset, CrisisMMDDataset_unpaired_random_image, CrisisMMDDataset_paired
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
import torch.nn.functional as F


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
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(
            outputs,
            batch["label"].to(device)
        )


            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    print(f"Eval Loss: {avg_loss:.4f}")
    print(f"Eval Accuracy: {accuracy:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "preds": all_preds,
        "labels": all_labels}

def kd_loss(student_logits, teacher_logits, T=4.0):
    """
    Logit distillation loss
    """
    return nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)


def feature_distill_one_epoch(student_model, teacher_model, projector ,data_loader,  optimizer,  device, lambda_fa=1.0):
    student_model.train()
    teacher_model.eval()
    projector.train()
    criterion = nn.CrossEntropyLoss()
    total = 0
    total_fa = 0.0
    total_ce = 0.0
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

    results =  {
        "loss": total_loss / len(data_loader),
        "ce": total_ce / len(data_loader),
        "fa": total_fa / len(data_loader),
        "acc": correct / total
    }
    
    return student_model, projector, optimizer, results

def logit_distill_one_epoch(student_model, teacher_model, data_loader,optimizer,  device):
    student_model.train()
    teacher_model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    total_fa = 0.0
    total_kl = 0.0
    total_loss = 0.0
    correct = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        with torch.no_grad():
            tea_logit = teacher_model(images) 

        
        stu_logits = student_model(input_ids, attention_mask)
        ### adding projector
      

        CE_loss = criterion(stu_logits, labels)
        KL_loss =  kd_loss(stu_logits, tea_logit)
        
        loss = CE_loss  + KL_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += CE_loss.item()
        total_kl += KL_loss.item()

        preds = torch.argmax(stu_logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    results =  {
        "loss": total_loss / len(data_loader),
        "ce": total_ce / len(data_loader),
        "fa": total_fa / len(data_loader),
        "acc": correct / total
    }
    
    return student_model, optimizer, results



def train(teacher = "bert"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    teacher_epochs = 20
    print("pre-training vision teacher with epochs: ",teacher_epochs)
    teacher_model = get_pretraining_techer_model(epochs= teacher_epochs)
    print("frozen teacher model...")
    teacher_model.eval()
    print("Init student model...")
    device = "cuda"
    student_model = SmallBertStudent().to(device)
    print("Load unpaired data...")
    train_dataset = CrisisMMDDataset_paired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= image_transform,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    val_dataset = CrisisMMDDataset_paired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= image_transform,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    test_dataset = CrisisMMDDataset_paired(csv_file= "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv",
                                              tokenizer= tokenizer,image_root="dataset/CrisisMMD_v2.0/",
                                              image_transform= image_transform,
                                              label_map= HUMANITARIAN_LABEL2ID)
    
    train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset,batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,batch_size=64, shuffle=True, num_workers=4)
    print("define the projector...")
   
    
    optimizer = torch.optim.AdamW(
        list(student_model.parameters()),
        lr=3e-4
    )
    student_epochs = 20
    for epoch in range(student_epochs):
        student_model, optimizer, results = feature_distill_one_epoch(student_model, teacher_model, train_loader, optimizer, device)
        print("training_result: ",results)
        evaluate(student_model, val_loader, device)

    print("Complete training, begin evaluation...")
    evaluate(student_model, test_loader, device)

if __name__ == "__main__":
    train()