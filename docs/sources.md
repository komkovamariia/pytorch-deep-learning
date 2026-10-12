# Course references and implementation scope

The TensorFlow examples in ML4HD were reimplemented with PyTorch; their saved Keras and HDF5 weights are not loaded. The course's seq2seq and Transformer topics are in [the NLP repository](https://github.com/komkovamariia/nlp), because their task is language translation.

| Source | Topics used |
| --- | --- |
| UniPd ML4HD: Lab_nn | Explicit gradients and dense classifiers |
| UniPd ML4HD: Lab_1 | PCA, compression and data preprocessing |
| UniPd ML4HD: Lab_2 and Lab_3 | CNN classification, autoencoders and multi-scale convolution |
| UniPd ML4HD: Lab_4 | Normalized graph aggregation, adapted from audio graphs to image grids |
| UniPd ML4HD: Lab_5 | Recurrent signal models and temporal preprocessing |
| UniPd ML4HD: Lab_6 | Spiking neurons and surrogate gradients |
| UniPd ML4HD: Lab_7 | Normal-only reconstruction and one-class anomaly detection |
| [HSE wiki, Deep Learning 2022](https://wiki.cs.hse.ru/Глубинное_обучение_2022) | Course scope: PyTorch, CV, NLP and image-to-image projects |
| [mryab/dl-hse-ami](https://github.com/mryab/dl-hse-ami) | Backpropagation, training, vision metric learning, segmentation, attention, adversarial examples and diffusion |
| [Practical_DL](https://github.com/yandexdataschool/Practical_DL) | PyTorch conventions, convolutional training and generative-model curriculum |
| [pytorch/examples](https://github.com/pytorch/examples) | Reference for concise model modules and explicit training scripts |

Periodic forecasting data are synthetic. The graph project preserves grid order in its readout. The spiking classifier has a spiking hidden layer and a continuous output layer. Generative models use small handwritten-digit images.

## Reference revisions

dl-hse-ami: `73bb1b5bd6c7af6b06abc52957de76efd4e91175`

practical-dl: `c073fe76f8330deb6f2f5bd7579ae75f4ed73958`
