# Course references and implementation scope

The supplied archives were inspected as reference material. The TensorFlow examples in ML4HD were reimplemented with PyTorch; their saved Keras and HDF5 weights are not loaded. The course's seq2seq and Transformer topics are in [the NLP repository](https://github.com/komkovamariia/nlp), because their task is language translation.

| Source | Topics used |
| --- | --- |
| UniPd Computer Vision: Image_Filtering_Notebook | Spatial filtering, gradients and Fourier filtering |
| UniPd Computer Vision: Lab3_SIFT and Logo Contest | Descriptor matching, ratio tests, geometry and image retrieval |
| UniPd Computer Vision: Lab 5 BoW | Train-only vocabulary fitting and histogram classification |
| UniPd Computer Vision: Lab6_MLP and Lab 7 CNN | Dense classifiers, LeNet, feature maps and transfer learning |
| UniPd ML4HD: Lab_nn | Explicit gradients and dense classifiers |
| UniPd ML4HD: Lab_1 | PCA, compression and data preprocessing |
| UniPd ML4HD: Lab_2 and Lab_3 | CNN classification, autoencoders and multi-scale convolution |
| UniPd ML4HD: Lab_4 | Normalized graph aggregation, adapted from audio graphs to image grids |
| UniPd ML4HD: Lab_5 | Recurrent signal models and temporal preprocessing |
| UniPd ML4HD: Lab_6 | Spiking neurons and surrogate gradients |
| UniPd ML4HD: Lab_7 | Normal-only reconstruction and one-class anomaly detection |
| [HSE wiki, Deep Learning 2022](https://wiki.cs.hse.ru/Глубинное_обучение_2022) | Course scope: PyTorch, CV, NLP and image-to-image projects; page read directly |
| [mryab/dl-hse-ami](https://github.com/mryab/dl-hse-ami) | Backpropagation, training, vision metric learning, segmentation, attention, adversarial examples and diffusion |
| [Practical_DL](https://github.com/yandexdataschool/Practical_DL) | PyTorch conventions, convolutional training and generative-model curriculum |
| [pytorch/examples](https://github.com/pytorch/examples) | Reference for concise model modules and explicit training scripts |
| [PyTorch transfer learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | Official weights, input transformations and frozen feature extractors |

SIFT's controlled warp is used for geometric error measurement. The cell segmentation and periodic forecasting data are synthetic. The Inception model is compact and does not reproduce full Inception-v4. The graph project preserves grid order in its readout. The spiking classifier has a spiking hidden layer and a continuous output layer. These distinctions are part of the experiments' scope.

Writing and code were reviewed using [avoid-ai-writing](https://github.com/conorbronsdon/avoid-ai-writing) and [clean-code-skills, Python track](https://github.com/ertugrul-dmr/clean-code-skills). These rules guide prose and code clarity. They are not a claim about authorship detection. Mathematical notation and documented model behavior are retained during cleanup.

## Reference revisions

dl-hse-ami: `73bb1b5bd6c7af6b06abc52957de76efd4e91175`

practical-dl: `c073fe76f8330deb6f2f5bd7579ae75f4ed73958`

avoid-ai-writing: `5a5cf6a45331384d4ff68f1f6ce50da24da0da4c`

clean-code-skills: `1b6b3cc1264b8fbe921c65002d05a3bf90ede178`
