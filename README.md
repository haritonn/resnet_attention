## About

This is an implementation of ResNet50 with channel attention mechanism 'from scratch'. The aim of the project is to demonstrate the practical implementation of channel attention in greater depth. Also here applied:
* Weight initialization (Kaiming);
* Early stopping;
* ... (see todo list)

## Training results

The launch was performed using the same parameters as in `config/config.py`.

<img width="1800" height="600" alt="image" src="https://github.com/user-attachments/assets/2b9244d9-3cf1-4e8f-bae0-9e832d8c690c" />

## Todo
- [ ] Inference (with plotting);
- [ ] Add dependecies;
- [ ] Attention plotting;
- [ ] Parameterize ResNet depth;
- [ ] Integration with MLFlow / ClearML;
- [ ] Metrics (F-score, confusion matrix);
- [ ] Comparison with default resnet;
- [ ] Optimization (ONNX);
- [ ] Add to readme after all.
