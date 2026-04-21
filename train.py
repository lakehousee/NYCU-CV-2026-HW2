import os, json, time, torch, random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# -------------------------
# CONFIG
# -------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NUM_CLASSES = 10
EPOCHS = 10

# -------------------------
# DATASET (same as notebook)
# -------------------------
class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        with open(ann_file) as f:
            data = json.load(f)

        self.images = data["images"]

        if not is_test:
            self.img_id2anns = {img["id"]: [] for img in self.images}
            for a in data["annotations"]:
                self.img_id2anns[a["image_id"]].append(a)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        img_id = info["id"]

        img = Image.open(os.path.join(self.img_dir, info["file_name"])).convert("RGB")
        w, h = img.size

        if self.is_test:
            if self.transform:
                img = self.transform(img)
            return img, img_id

        boxes, labels = [], []
        for a in self.img_id2anns[img_id]:
            x, y, bw, bh = a["bbox"]
            boxes.append([(x+bw/2)/w, (y+bh/2)/h, bw/w, bh/h])
            labels.append(a["category_id"] - 1)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

        if self.transform:
            img = self.transform(img)

        return img, target


# -------------------------
# TRANSFORMS
# -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------------
# MODEL (DETR)
# -------------------------
import torch.hub
import torch.nn as nn

detr_pretrained = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)

model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=False)
model.class_embed = nn.Linear(256, NUM_CLASSES + 1)

# load backbone weights
sd_pre = detr_pretrained.state_dict()
sd = model.state_dict()
for k,v in sd_pre.items():
    if k in sd and sd[k].shape == v.shape:
        sd[k] = v
model.load_state_dict(sd)

model = model.to(DEVICE)
del detr_pretrained

# -------------------------
# OPTIMIZER
# -------------------------
params = [
    {"params": model.parameters(), "lr": 1e-4}
]
optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

# -------------------------
# TRAIN LOOP (simplified placeholder)
# -------------------------
def train_one_epoch(loader):
    model.train()
    total_loss = 0

    for imgs, targets in loader:
        imgs = torch.stack(imgs).to(DEVICE)

        outputs = model(imgs)

        loss = outputs["pred_logits"].mean()  # placeholder loss (use your real loss!)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# RUN TRAINING
# -------------------------
for epoch in range(EPOCHS):
    loss = train_one_epoch(train_loader)
    print(f"Epoch {epoch}: loss {loss:.4f}")

    torch.save({
        "model": model.state_dict(),
        "epoch": epoch
    }, f"{CHECKPOINT_DIR}/last.pth")