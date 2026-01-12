from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from MMDDataset import CrisisMMDTextDataset
def train():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    csv_file = "../dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv"
    df = pd.read_csv(csv_file, sep="\t")
    print("len df: ", len(df))
    train_dataset = CrisisMMDTextDataset(
    csv_file="../dataset/CrisisMMD_v2.0/crisismmd_datasplit_all/task_informative_text_img_train.tsv",
    tokenizer=tokenizer
)
    print("len dataset: ",len(train_dataset))  
#     dev_dataset = CrisisMMDTextDataset(
#     tsv_file="annotations/task1_dev_text.tsv",
#     tokenizer=tokenizer
# )

#     train_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True,
#     num_workers=4
# )

#     dev_loader = DataLoader(
#     dev_dataset,
#     batch_size=32,
#     shuffle=False
# )

if __name__ == "__main__":
    print("start test")
    train()