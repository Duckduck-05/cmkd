from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from utils.MMDDataset import CrisisMMDTextDataset
import torch.nn as nn
from torch.optim import AdamW
import torch 
from model.Bert_based import BertClassifier, BertModel

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
        "labels": all_labels
    }

def train():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    label_map = {
    "not_informative": 0,
    "informative": 1
}
    csv_file = "dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv"
    df = pd.read_csv(csv_file, sep="\t")
    print("len df: ", len(df))
    train_dataset = CrisisMMDTextDataset(
    csv_file="dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv",
    tokenizer=tokenizer, 
    label_map= label_map
)
    print("len dataset: ",len(train_dataset))  
    dev_dataset = CrisisMMDTextDataset(
    csv_file="dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_dev.tsv",
    tokenizer=tokenizer, 
    label_map= label_map
)
    
    test_dataset = CrisisMMDTextDataset(
    csv_file="dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_test.tsv",
    tokenizer=tokenizer, 
    label_map= label_map
)
    train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

    dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False
)
    test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)
    print("complete data loader")
    device = "cuda" 

    model = BertClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    print("start traing")
    evaluate(model,dev_loader, device)
    for epoch in range(3):
       model.train()
       total_loss = 0

       for batch in train_loader:
           optimizer.zero_grad()

           logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device)
        )

           loss = criterion(
            logits,
            batch["label"].to(device)
        )

           loss.backward()
           optimizer.step()
           total_loss += loss.item()
       
       
       print(f"Epoch {epoch} | Train loss: {total_loss / len(train_loader):.4f}")
       evaluate(model,dev_loader, device)
    
    print("finish training, start testing...")
    evaluate(model,test_loader, device)

if __name__ == "__main__":
    print("start test")
    train()