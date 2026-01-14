from torchvision import transforms
import torch 
import torch.nn as nn
from utils.MMDDataset import CrisisMMDHumanitarianImageDataset
from model.MobileNet import MobileNetV2Classifier, MobileNetV2Humanitarian, MobileNetV2Student
from torch.utils.data import DataLoader
import torch.optim as optim




image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



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

LABEL2ID = {l: i for i, l in enumerate(HUMANITARIAN_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NUM_LABELS = 8

def evaluate_image(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total



def train_image_epoch(model, dataloader, optimizer,  device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print("begin load dataset...")
    train_dataset = CrisisMMDHumanitarianImageDataset(tsv_file="dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_train.tsv",
                                                     label2id= LABEL2ID,
                                                     transform= image_transform,
                                                     image_root="dataset/CrisisMMD_v2.0/")
    print("len train loader: ", len(train_dataset))
    val_dataset = CrisisMMDHumanitarianImageDataset(tsv_file="dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_dev.tsv",
                                                     label2id= LABEL2ID,
                                                     transform= image_transform,
                                                     image_root="dataset/CrisisMMD_v2.0/")
    
    test_dataset = CrisisMMDHumanitarianImageDataset(tsv_file="dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_humanitarian_text_img_test.tsv",
                                                     label2id= LABEL2ID,
                                                     transform= image_transform,
                                                     image_root="dataset/CrisisMMD_v2.0/")
    

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print("init model...")
    device = "cuda"
    model = MobileNetV2Student().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    print("student model params: ", count_trainable_parameters(model))
    epochs = 60
    print("start training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_image_epoch(
            model, train_loader, optimizer, device
        )

        val_loss, val_acc = evaluate_image(
            model, val_loader, device
        )

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

    print("evaluate on test set")
    test_loss, test_acc = evaluate_image(
            model, test_loader, device
        )

    print(
            f"test Loss: {val_loss:.4f}, test Acc: {val_acc:.4f}"
        )


if __name__== "__main__":
    main()