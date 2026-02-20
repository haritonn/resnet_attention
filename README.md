## About

This is an implementation of ResNet50 with channel attention mechanism 'from scratch'. The aim of the project is to demonstrate the practical implementation of channel attention in greater depth. Also here applied:
* Weight initialization (Kaiming);
* Early stopping;
* Clearml integration;
* Attention visualization;
* Comparison with default resnet;
* ~~ONNX optimization~~ maybe later :)).

## Launching

Requirements:
* python (ofc)
* uv package manager

### Installtion commands
```bash
git clone git@github.com:haritonn/resnet_attention.git
cd resnet_attention
uv sync
```

### Running commands
```bash
uv run train.py # training
uv run inference.py # inference

# train both models
chmod +x ./train_both.sh
./train_both
```

### Running arguments
```txt
--model_type - attention or baseline training
--clearml-name - naming of project in clearml
--checkpoints - folder for checkpoints storage
--plots - folder for plots storage
```

## Training results

The launch was performed using the same parameters as in `config/config.py`.
### Accuracy
![Training Accuracy](imgs/Train_acc.png)
### F-score
![Training F-score](imgs/Train_fscore.png)
### Loss (CrossEntropyLoss)
![Training Loss](imgs/Train_loss.png)

## Todo
- [x] Inference (with plotting);
- [x] Add dependecies;
- [x] Attention plotting;
- [ ] ~~Parameterize ResNet depth;~~
- [x] Integration with MLFlow / ClearML;
- [x] Metrics (F-score, confusion matrix);
- [x] Comparison with default resnet;
- [ ] Optimization (ONNX); (maybe later)
- [x] Add to readme after all.
