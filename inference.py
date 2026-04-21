import os, json, torch, zipfile
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# PATHS
# -------------------------
TEST_DIR = "./test"
CKPT_PATH = "./checkpoints/best_phase1.pth"

# -------------------------
# MODEL
# -------------------------
import torch.hub
import torch.nn as nn

NUM_CLASSES = 10

model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=False)
model.class_embed = nn.Linear(256, NUM_CLASSES + 1)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model.load_state_dict(ckpt["model"])

model = model.to(DEVICE)
model.eval()

print(f"Loaded model epoch {ckpt['epoch']}")

# -------------------------
# TRANSFORM
# -------------------------
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -------------------------
# TEST FILES
# -------------------------
files = sorted(os.listdir(TEST_DIR))

predictions = []

with torch.no_grad():
    for i, f in enumerate(files):
        img = Image.open(os.path.join(TEST_DIR, f)).convert("RGB")
        w,h = img.size

        x = tf(img).unsqueeze(0).to(DEVICE)

        out = model(x)

        prob = out["pred_logits"].softmax(-1)[0]
        boxes = out["pred_boxes"][0]

        scores, labels = prob[:,:-1].max(-1)
        keep = scores > 0.3

        for s,l,b in zip(scores[keep], labels[keep], boxes[keep]):
            cx,cy,bw,bh = b.tolist()

            predictions.append({
                "image_id": i+1,
                "category_id": int(l)+1,
                "bbox": [(cx-bw/2)*w, (cy-bh/2)*h, bw*w, bh*h],
                "score": float(s)
            })

print("preds:", len(predictions))

with open("pred.json", "w") as f:
    json.dump(predictions, f)

with zipfile.ZipFile("submission.zip", "w") as z:
    z.write("pred.json", "pred.json")

print("submission ready")