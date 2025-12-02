# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:21:18 2025

@author: Manisha
"""

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image

# device, model, optimizer, criterion assumed already defined
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2️⃣ Transforms (preprocessing)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),       # ResNet input size
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, saturation=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3️⃣ Dataset
dataset = datasets.Caltech101(root='.', download=True, transform=train_transform)

# 4️⃣ Train/Val split
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Apply validation transforms to val_dataset
val_dataset.dataset.transform = val_transform

class_names = dataset.categories   
num_classes = len(class_names)
                  
# 5️⃣ DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Replace the final FC layer for your number of classes
num_classes = 101  # Caltech101 has 101 classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Fine tune all layers
for param in model.parameters():
  param.requires_grad = True

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Early stopping parameters
early_stop_patience = 5
best_val_loss = float('inf')
best_model_wts = copy.deepcopy(model.state_dict())
epochs_no_improve = 0

def clean_labels(labels, outputs=None):
    """
    Normalize labels to be a 1D LongTensor of shape [batch_size].
    If labels look like one-hot, convert using argmax along dim=1.
    """
    # If tuple/list (some datasets return (target, meta))
    if isinstance(labels, (list, tuple)):
        # try to extract the first element
        labels = labels[0]

    # If it's already a tensor, proceed; otherwise coerce
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Move to CPU for shape ops (we'll .to(device) later)
    labels = labels.detach().cpu()

    # If labels are one-hot (N, C) and outputs provided, convert to indices
    if labels.dim() == 2:
        if outputs is not None and outputs.dim() == 2 and labels.size(1) == outputs.size(1):
            # assume one-hot -> convert
            labels = labels.argmax(dim=1)
        elif labels.size(1) == 1:
            labels = labels.squeeze(1)
        else:
            # Unexpected multi-target; keep as-is to surface error later
            pass

    # If labels have extra dims e.g., (N,1,1) -> squeeze
    if labels.dim() > 1:
        labels = labels.squeeze()

    # Final safety: ensure 1D
    if labels.dim() == 0:
        labels = labels.unsqueeze(0)

    # Make integer type required by CrossEntropy
    labels = labels.long()

    return labels

# Training + Validation with robust label handling and diagnostics
num_epochs = 20
seen_bad_batch = False  # only print detailed diagnostic once
# After each epoch, store metrics
train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(num_epochs):
        
    # ---------- TRAIN ----------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        # CLEAN LABELS
        labels = clean_labels(labels, outputs=None)  # outputs not available yet in train
        # OPTIONAL: print problematic batch only once
        if (labels.dim() != 1 or labels.dtype != torch.int64) and not seen_bad_batch:
            print("Train: after cleaning, labels shape/dtype:", labels.shape, labels.dtype)
            seen_bad_batch = True

        # Move to device
        images = images.to(device)
        labels = labels.to(device)

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(images)

        # FINAL ASSERTIONS BEFORE LOSS
        # outputs: [N, C]; labels: [N]
        if outputs.dim() != 2:
            raise RuntimeError(f"Unexpected outputs shape {outputs.shape} (expected [N, C])")
        if labels.dim() != 1:
            # try one-more-attempt: maybe labels are one-hot and match outputs
            if labels.dim() == 2 and labels.size(1) == outputs.size(1):
                labels = labels.argmax(dim=1).to(device)
            else:
                # print debug and skip this batch to avoid crash
                print(f"Skipping train batch {batch_idx} due to labels dim {labels.shape}")
                continue

        # Shape check
        if labels.size(0) != outputs.size(0):
            print(f"Skipping train batch {batch_idx} due to batch-size mismatch: outputs {outputs.size(0)}, labels {labels.size(0)}")
            continue

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * outputs.size(0)
        _, preds = outputs.max(1)
        train_total += labels.size(0)
        train_correct += preds.eq(labels).sum().item()

    avg_train_loss = train_loss / (train_total if train_total>0 else 1)
    train_acc = 100.0 * train_correct / (train_total if train_total>0 else 1)

    # ---------- VALIDATION ----------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            labels = clean_labels(labels, outputs=None)
            # Move to device
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # If labels somehow equal one-hot matching outputs columns, convert
            if labels.dim() == 2 and labels.size(1) == outputs.size(1):
                labels = labels.argmax(dim=1)

            # If still not 1D, print diagnostic and skip
            if labels.dim() != 1:
                if not seen_bad_batch:
                    print("Val: problematic labels found:", labels.shape, labels.dtype)
                    print(labels)
                    seen_bad_batch = True
                print(f"Skipping val batch {batch_idx} due to labels dim {labels.shape}")
                continue

            # Shape check
            if labels.size(0) != outputs.size(0):
                print(f"Skipping val batch {batch_idx} due to batch-size mismatch: outputs {outputs.size(0)}, labels {labels.size(0)}")
                continue

            loss = criterion(outputs, labels)
            val_loss += loss.item() * outputs.size(0)
            _, preds = outputs.max(1)
            val_total += labels.size(0)
            val_correct += preds.eq(labels).sum().item()

    avg_val_loss = val_loss / (val_total if val_total>0 else 1)
    val_acc = 100.0 * val_correct / (val_total if val_total>0 else 1)

    # In the training loop, after computing losses and accuracies:
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    scheduler.step(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} — Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.2f}% | Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      best_model_wts = copy.deepcopy(model.state_dict())
      epochs_no_improve = 0
    else:
      epochs_no_improve += 1
      if epochs_no_improve >= early_stop_patience:
        print(f"Early stops triggered at {epoch+1} epochs")
        break

# Load the best model
model.load_state_dict(best_model_wts)
print(f"Training completed, Best validation loss {avg_val_loss:.2f}")


# After training, plot
plt.figure(figsize=(12,5))

# Loss plot
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

# Accuracy plot
plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy per Epoch')
plt.legend()

plt.show()

# Grad-CAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (x.size(2), x.size(3)))
        return cam

# Example usage
# model: your trained ResNet18
# target_layer: model.layer4[-1].conv2
gradcam = GradCAM(model, model.layer4[-1].conv2)

# Get one image
image, label = val_dataset[0]
input_tensor = image.unsqueeze(0).to(device)

# Generate CAM
mask = gradcam.generate(input_tensor, class_idx=label)

# Overlay on original image
img = image.permute(1,2,0).cpu().numpy()
img = (img - img.min()) / (img.max()-img.min())
heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
heatmap = heatmap/255.0
overlay = 0.5*img + 0.5*heatmap
plt.imshow(overlay)
plt.axis('off')
plt.show()



# Put model in eval mode
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- Classification Report ---
print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap="viridis", square=True, cbar=True)
plt.title("Confusion Matrix — Caltech101")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# Pick last conv layer of ResNet18
target_layer = model.layer4[-1].conv2

# store gradients & activations
gradients = None
activations = None

def save_backward(grads):
    global gradients
    gradients = grads

def save_forward(act):
    global activations
    activations = act

# register hooks
target_layer.register_forward_hook(lambda m, i, o: save_forward(o))
target_layer.register_backward_hook(lambda m, gi, go: save_backward(go[0]))

def generate_gradcam(img_tensor):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    output = model(img_tensor)
    class_idx = output.argmax(dim=1).item()

    # Backprop for chosen class
    model.zero_grad()
    output[0, class_idx].backward()

    # Grad-CAM heatmap
    grads = gradients.cpu().data.numpy()[0]
    acts = activations.cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for w, a in zip(weights, acts):
        cam += w * a

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    cam = cv2.resize(cam, (224, 224))
    cam = np.uint8(255 * cam)

    return cam, class_idx

def show_gradcam(image, heatmap):
    image = image.permute(1,2,0).cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (0.4 * image + 0.6 * heatmap/255.0)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()
    
sample_img, _ = val_dataset[0]
cam, pred = generate_gradcam(sample_img)
show_gradcam(sample_img, cam)

def gradcam_plus_plus(activations, gradients):
    grads2 = gradients ** 2
    grads3 = gradients ** 3

    denom = 2 * grads2 + np.sum(activations * grads3, axis=(1,2), keepdims=True)
    a = grads2 / (denom + 1e-7)

    weights = np.sum(a * np.maximum(gradients, 0), axis=(1,2))
    cam = np.sum(weights[:, None, None] * activations, axis=0)
    cam = np.maximum(cam, 0)

    cam = cv2.resize(cam, (224,224))
    cam = cam / cam.max()
    cam = np.uint8(255 * cam)

    return cam

def generate_gradcam_pp(img_tensor):
    model.eval()
    img_tensor = img_tensor.unsqueeze(0).to(device)

    output = model(img_tensor)
    class_idx = output.argmax(dim=1).item()

    model.zero_grad()
    output[0, class_idx].backward()

    acts = activations.cpu().data.numpy()[0]
    grads = gradients.cpu().data.numpy()[0]

    cam_pp = gradcam_plus_plus(acts, grads)
    return cam_pp, class_idx

cam_pp, pred = generate_gradcam_pp(sample_img)
show_gradcam(sample_img, cam_pp)

torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': class_names,
}, "caltech101_resnet_best.pth")

print("Model saved!")


