# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from PIL import Image
# from tqdm import tqdm
# from transformers import ViTForImageClassification, ViTFeatureExtractor

# # ====== Config ======
# BATCH_SIZE = 256
# EPOCHS = 30
# LR = 0.00005
# NUM_CLASSES = 200
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # ====== Load HF Feature Extractor ======
# extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

# # ====== Dataset Wrappers ======
# class TinyImageNetHF(torch.utils.data.Dataset):
#     def __init__(self, hf_data, extractor):
#         self.data = hf_data
#         self.extractor = extractor
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         img = self.data[idx]['image'].convert("RGB")
#         label = self.data[idx]['label']
#         inputs = self.extractor(images=img, return_tensors="pt")
#         pixel_values = inputs['pixel_values'].squeeze(0)
#         return pixel_values, label

# # ====== Load Dataset ======
# dataset = load_dataset("zh-plus/tiny-imagenet")

# train_dataset = TinyImageNetHF(dataset['train'], extractor)
# val_dataset = TinyImageNetHF(dataset['valid'], extractor)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# # ====== Model ======
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=1000)
# # print(model)
# model.classifier = nn.Linear(768, NUM_CLASSES, bias=True)
# model.to(DEVICE)

# # ====== Loss and Optimizer ======
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=LR)

# # ====== Training ======
# def train():
#     for epoch in range(EPOCHS):
#         model.train()
#         total_loss, correct, total = 0, 0, 0
#         loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
#         for images, labels in loop:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(pixel_values=images)
#             logits = outputs.logits
#             loss = criterion(logits, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             _, preds = torch.max(logits, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#             loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

#         validate()
#         torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

# # ====== Validation ======
# def validate():
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(pixel_values=images)
#             logits = outputs.logits
#             _, preds = torch.max(logits, 1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# # ====== Run ======
# if __name__ == "__main__":
#     train()


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ====== Config ======
BATCH_SIZE = 512
EPOCHS = 30
LR = 0.00005
NUM_CLASSES = 200
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ====== Transform ======
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)),
])

# ====== Load Hugging Face Dataset ======
dataset = load_dataset("zh-plus/tiny-imagenet")

# ====== Dataset Wrappers ======
class ImageNetHF(torch.utils.data.Dataset):
    def __init__(self, hf_data, transform):
        self.data = hf_data
        self.transform = transform
        self.label2id = {label: i for i, label in enumerate(sorted(set(self.data['label'])))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]['image']
        label = self.data[idx]['label']
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

train_dataset = ImageNetHF(dataset['train'], transform)
val_dataset = ImageNetHF(dataset['valid'], valid_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ====== Model ======
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
state_dict = model.state_dict()
for name, param in state_dict.items():
    if 'fc' not in name:
        param.requires_grad=False
    else:
        param.requires_grad = True

model.load_state_dict(state_dict)
model.to(DEVICE)


# ====== Loss and Optimizer ======
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ====== Training ======
def train():
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)
        for images, labels in loop:
            # print(images.shape)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        validate()
        torch.save(model.state_dict(), f"model_epoch_resnet50_epoch_{epoch+1}.pth")

# ====== Validation ======
def validate():
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# ====== Run ======
if __name__ == "__main__":
    train()