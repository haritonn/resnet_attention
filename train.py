import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score, confusion_matrix
from torchvision.models import resnet50
from matplotlib import pyplot as plt
from clearml import Task
import argparse
import pickle
import tqdm
import os

import config.config as config
from utils.utils import EarlyStopping
from dataset.dataset import CifarDataset
from models.resnet50 import ResNet

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, choices=['attention', 'baseline'],
                    default='attention', help='Model type: with attention or default.')

args = parser.parse_args()

task = Task.init(
    project_name="Resnet_attention_comparison",
    task_name=f"Resnet_{args.model_type}",
    tags=[args.model_type, 'cifar100']
)

task.connect_configuration(configuration=config.__dict__, name='training_config')
task.set_parameter('model_type', args.model_type)

# Loading data, making train/val split
all_data = CifarDataset(config.TRAIN_DIR, is_train=config.USING_AUGMENTATIONS)
train_data, val_data = random_split(
    all_data,
    [1 - config.VAL_SPLIT, config.VAL_SPLIT],
    torch.Generator().manual_seed(config.RANDOM_SEED),
)

train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=False)

# Core objects init: model, early stopping, criterion...
if args.model_type == 'attention':
    model = ResNet(in_channels=3, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    print("Using ResNet with Channel Attention")
else:
    model = resnet50(weights=None, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    print("Using Baseline ResNet50")

earlystop = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE)
criterion = getattr(torch.nn, config.CRITERION)()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)

total_params = sum(p.numel() for p in model.parameters())
task.get_logger().report_single_value('total params', total_params)
print(f'Total parameters: {total_params}')

# For future plotting
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

# Training process
for epoch in range(1, config.NUM_EPOCHS + 1):
    torch.cuda.empty_cache()

    model.train()
    model.zero_grad()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    all_predictions_train = []
    all_labels_train = []
    pbar_train = tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")

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

        all_predictions_train.extend(predicted.cpu().numpy())
        all_labels_train.extend(labels.cpu().numpy())

        pbar_train.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{100 * train_correct / train_total:.2f}%",
            }
        )

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = 100 * train_correct / train_total

    all_predictions_val = []
    all_labels_val = []

    # Early stopping checking
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    pbar_val = tqdm.tqdm(
        val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Val]", unit="batch"
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
            all_predictions_val.extend(predicted.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())

            pbar_val.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * val_correct / val_total} %",
                }
            )

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100 * val_correct / val_total

    history["train_loss"].append(avg_train_loss)
    history["val_loss"].append(avg_val_loss)
    history["train_acc"].append(avg_train_acc)
    history["val_acc"].append(avg_val_acc)

    f_train = f1_score(all_labels_train, all_predictions_train, average='weighted')
    f_val = f1_score(all_labels_val, all_predictions_val, average='weighted')

    logger = task.get_logger()
    # Loss
    logger.report_scalar(
        title='Loss',
        series='Train',
        value=avg_train_loss,
        iteration=epoch
    )
    logger.report_scalar(
        title='Loss',
        series='Val',
        value=avg_val_loss,
        iteration=epoch
    )
    # Accuracy
    logger.report_scalar(
        title='Accuracy',
        series='Train',
        value=avg_train_acc,
        iteration=epoch
    )
    logger.report_scalar(
        title='Accuracy',
        series='Val',
        value=avg_val_acc,
        iteration=epoch
    )
    # f1_score
    logger.report_scalar(
        title='F-score',
        series='Train',
        value=f_train,
        iteration=epoch
    )
    logger.report_scalar(
        title='F-score',
        series='Val',
        value=f_val,
        iteration=epoch
    )

    cm = confusion_matrix(all_labels_val, all_predictions_val)
    logger.report_confusion_matrix(
        title='Confusion Matrix', 
        series='Validation',  
        matrix=cm, 
        iteration=epoch
    )
    # If early stopping triggered - using best model at this point
    # and exiting training process.
    should_continue, best_state = earlystop.step(avg_val_loss, model)
    if not should_continue:
        print("Eearly stopping triggered!")
        model.load_state_dict(best_state)
        break


best_acc = max(history['val_acc'])
task.get_logger().report_single_value('Best validation accuracy', best_acc)

# Saving checkpoints 
os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
model_path = f"{config.CHECKPOINTS_DIR}/resnet50_{args.model_type}.pth"
torch.save(model.state_dict(), model_path)
print(f"Training complete. Model saved to {config.CHECKPOINTS_DIR}")

task.upload_artifact(f'model_weights_{args.model_type}', model_path)


# Plotting results
os.makedirs(config.TRAIN_RESULTS_DIR, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot losses
ax1.plot(history["train_loss"], label="Train Loss")
ax1.plot(history["val_loss"], label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True)

# Plot accuracies
ax2.plot(history["train_acc"], label="Train Acc")
ax2.plot(history["val_acc"], label="Val Acc")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()
ax2.grid(True)

# Saving results
plt.tight_layout()
plotting_path = f"{config.TRAIN_RESULTS_DIR}/training_{args.model_type}.png"
plt.savefig(plotting_path, dpi=150)

task.get_logger().report_matplotlib_figure(
    title='Training History',
    series=f'{args.model_type} Model',
    figure=fig,
    report_interactive=False
)
