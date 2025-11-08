## About

This is an implementation of ResNet50 with channel attention mechanism 'from scratch'. The aim of the project is to demonstrate the practical implementation of channel attention in greater depth. Also here applied:
* Weight initialization (Kaiming);
* Early stopping;
* ... (see todo list)

## Training results

The launch was performed using the same parameters as in `config/config.py`.

![Training results](https://github.com/haritonn/resnet_attention/tree/master/train_results)

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
