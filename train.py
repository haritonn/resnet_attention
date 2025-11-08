import torch
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt
import pickle
import tqdm
import os

import config.config as config
from utils.utils import EarlyStopping
from dataset.dataset import CifarDataset
from models.resnet50 import ResNet

all_data = CifarDataset(config.TRAIN_DIR, is_train=config.USING_AUGMENTATIONS)
train_data, val_data = random_split( all_data, [1 - config.VAL_SPLIT, config.VAL_SPLIT]
                                   , torch.Generator().manual_seed(config.RANDOM_SEED) )

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)

model = ResNet(in_channels=3, num_classes=config.NUM_CLASSES).to(config.DEVICE)
earlystop = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)

criterion = getattr(torch.nn, config.CRITERION)()
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

for epoch in range(1, config.NUM_EPOCHS+1):
    torch.cuda.empty_cache()

    model.train()
    model.zero_grad()
    train_loss = 0.0
    train_correct = 0
    train_total = 0 
    pbar_train = tqdm.tqdm(train_loader, desc=f'Epoch {epoch} [Train]', unit='batch')

    for batch, labels in pbar_train:
        batch, labels = batch.to(config.DEVICE), labels.to(config.DEVICE)
        outputs = model(batch)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)  
        train_correct += (predicted == labels).sum().item()

        pbar_train.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * train_correct / train_total:.2f}%'  
        })
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = 100 * train_correct / train_total

    # Early stopping checking
    model.eval()
    val_loss = .0
    val_correct = 0
    val_total = 0

    pbar_val = tqdm.tqdm(
        val_loader, 
        desc=f'Epoch {epoch}/{config.NUM_EPOCHS} [Val]', 
        unit='batch'
    )

    with torch.no_grad():
        for batch, labels in pbar_val:
            batch, labels = batch.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(batch)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            pbar_val.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * val_correct / val_total} %'
            })

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100 * val_correct / val_total

    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_acc'].append(avg_train_acc)
    history['val_acc'].append(avg_val_acc)

    should_continue, best_state = earlystop.step(avg_val_loss, model)
    if not should_continue:
        print('Eearly stopping triggered!')
        model.load_state_dict(best_state)


os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
torch.save(model.state_dict(), f'{config.CHECKPOINTS_DIR}/resnet50_cifar100.pth')
print(f'Training complete. Model saved to {config.CHECKPOINTS_DIR}')


os.makedirs(config.TRAIN_RESULTS_DIR, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot losses
ax1.plot(history['train_loss'], label='Train Loss')
ax1.plot(history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Plot accuracies
ax2.plot(history['train_acc'], label='Train Acc')
ax2.plot(history['val_acc'], label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(f'{config.TRAIN_RESULTS_DIR}/training_history.png', dpi=150)
