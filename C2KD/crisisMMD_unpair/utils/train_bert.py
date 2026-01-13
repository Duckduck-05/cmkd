from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from MMDDataset import CrisisMMDTextDataset,CrisisMMDHumanitarianTextDataset
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification
from model.Bert_based import BertClassifier, BertModel

HUMANITARIAN_LABELS = [
    "affected_individuals",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "missing_or_found_people",
    "not_humanitarian",
    "other_relevant_information",
    "rescue_volunteering_or_donation_effort",
    "vehicle_damage",
]
LABEL2ID = {label: idx for idx, label in enumerate(HUMANITARIAN_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

def normalize_label(label: str):
    if not isinstance(label, str):
        return None
    return (
        label.lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )



def train():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    csv_file = "../dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv"
    df = pd.read_csv(csv_file, sep="\t")
    print("len df: ", len(df))
    train_dataset = CrisisMMDHumanitarianTextDataset(
    csv_file="../dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv",
    tokenizer=tokenizer, 
    label2id= LABEL2ID
)
    print("len dataset: ",len(train_dataset))  
    dev_dataset = CrisisMMDHumanitarianTextDataset(
    csv_file="../dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_dev.tsv",
    tokenizer=tokenizer, 
    label2id= LABEL2ID
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
    print("complete data loader")
    device = "cuda" 
    model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID
    )
    #model = BertClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
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

if __name__ == "__main__":
    print("start test")
    train()