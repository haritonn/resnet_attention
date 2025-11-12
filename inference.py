import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from clearml import Task
import tqdm
import argparse
import os
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F

import config.config as config
from dataset.dataset import CifarDataset
from models.resnet50 import ResNet

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_type",
    type=str,
    choices=["attention", "baseline"],
    default="attention",
    help="Model type: with attention or default.",
)
parser.add_argument(
    "--clearml-name",
    type=str,
    default="Resnet_attention_comparison",
    help="Naming of project in ClearML.",
)
parser.add_argument(
    "--plots",
    type=str,
    default=config.TRAIN_RESULTS_DIR,
    help="Folder for training plots.",
)

parser.add_argument(
    "--checkpoints",
    type=str,
    default=config.CHECKPOINTS_DIR,
    help="Folder for checkpoints to use.",
)
args = parser.parse_args()

task = Task.init(
    project_name=args.clearml_name, task_name=f"Inference_{args.model_type}"
)

task.connect_configuration(configuration=config.__dict__, name="inference_config")
task.set_parameter("model_type", args.model_type)

test_data = CifarDataset(data_path=config.TEST_DIR, is_train=False)
test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)

class_names = (
    test_data.classes
    if hasattr(test_data, "classes")
    else [f"Class_{i}" for i in range(config.NUM_CLASSES)]
)

if args.model_type == "attention":
    model = ResNet(in_channels=3, num_classes=config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(
        f"{args.checkpoints}/resnet50_{args.model_type}.pth", map_location=config.DEVICE
    )
    model.load_state_dict(checkpoint)
    print("Using ResNet with Channel Attention")
else:
    model = resnet50(
        weights=ResNet50_Weights.DEFAULT, num_classes=config.NUM_CLASSES
    ).to(config.DEVICE)
    print("Using Baseline ResNet50")

total_params = sum(p.numel() for p in model.parameters())
task.get_logger().report_single_value("total params", total_params)

model.eval()

all_predictions = []
all_labels = []
all_probs = []
total_loss = 0.0
correct = 0
total = 0

criterion = torch.nn.CrossEntropyLoss()

with torch.no_grad():
    for idx, (images, labels) in enumerate(tqdm.tqdm(test_loader, desc="Inference")):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

avg_loss = total_loss / len(test_loader)
accuracy = 100 * correct / total

logger = task.get_logger()
logger.report_single_value("test_loss", avg_loss)
logger.report_single_value("test_acc", accuracy)

cm = confusion_matrix(all_labels, all_predictions)
logger.report_confusion_matrix(
    title="Confusion Matrix",
    series=args.model_type,
    matrix=cm,
    iteration=0,
    xaxis="Predicted",
    yaxis="True",
    xlabels=class_names,
    ylabels=class_names,
)
print(f"\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=class_names))

plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - {args.model_type}")
plt.colorbar()

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.tight_layout()

os.makedirs(args.plots, exist_ok=True)
plot_path = os.path.join(args.plots, f"confusion_matrix_{args.model_type}.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nConfusion matrix saved to: {plot_path}")

logger.report_image(
    "confusion_matrix", f"{args.model_type}", iteration=0, image=plt.gcf()
)
