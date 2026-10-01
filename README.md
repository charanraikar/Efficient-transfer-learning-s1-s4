# Efficient Transfer Learning: Optimal Layer-Freezing Strategies

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/charanraikar/Efficient-transfer-learning-s1-s4/blob/main/efficient_transfer_learning%28s1_s4%29.ipynb)

This project investigates how different layer-freezing strategies affect the accuracy, training time, and GPU-memory requirements of transfer learning for scene image classification.

A pretrained ResNet-18 model is evaluated using four fine-tuning strategies on the Intel Image Classification dataset. The experiments measure test accuracy, macro-F1 score, epoch time, total training time, and peak GPU memory.

## Research Objective

The project addresses the following question:

> How many layers of a pretrained convolutional neural network should be unfrozen to obtain the best balance between classification accuracy and computational efficiency?

Instead of choosing only between fully frozen and fully fine-tuned models, this project compares different levels of selective unfreezing.

## Layer-Freezing Strategies

| Strategy | Trainable components | Trainable parameters |
|---|---|---:|
| S1 - Head Only | Classification head | 3,078 (0.03%) |
| S2 - Last Block | ResNet Layer 4 and classification head | 8,396,806 (75.1%) |
| S3 - Last Two Blocks | ResNet Layers 3-4 and classification head | 10,496,518 (93.9%) |
| S4 - Full Fine-Tuning | Complete ResNet-18 model | 11,179,590 (100%) |

## Dataset

The project uses the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), containing 17,034 RGB images across six scene categories:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

### Dataset Split

| Split | Number of images |
|---|---:|
| Training | 11,929 |
| Validation | 2,105 |
| Testing | 3,000 |
| Total | 17,034 |

Training images are augmented using horizontal flipping, rotation, and colour jitter. All images are resized to 224 × 224 pixels and normalized using ImageNet statistics.

## Experimental Setup

- Backbone: ResNet-18 pretrained on ImageNet-1K
- Optimizer: Adam
- Learning rate: `0.001`
- Weight decay: `0.0001`
- Batch size: `32`
- Maximum epochs: `20`
- Early-stopping patience: `4`
- Learning-rate scheduler: StepLR
- Loss function: Cross-entropy loss
- Random seed: `42`
- Hardware: NVIDIA Tesla T4 GPU with 15.6 GB VRAM
- Environment: Google Colab

## Results

| Strategy | Test Accuracy | Macro-F1 | Time/Epoch | Total Time | Peak GPU Memory | Epochs |
|---|---:|---:|---:|---:|---:|---:|
| S1 - Head Only | 90.43% | 0.9059 | 50.3 s | 1005.7 s | 379 MiB | 20 |
| **S2 - Last Block** | **92.40%** | 0.9251 | **50.2 s** | **250.8 s** | 475 MiB | 5 |
| S3 - Last Two Blocks | 92.20% | 0.9236 | 51.4 s | 770.8 s | 501 MiB | 15 |
| S4 - Full Fine-Tuning | 92.37% | **0.9256** | 57.3 s | 1145.1 s | 991 MiB | 20 |

## Key Findings

- S2 achieved the highest test accuracy of **92.40%**.
- S2 completed training approximately **4.6 times faster** than full fine-tuning.
- S2 required approximately **52% less peak GPU memory** than S4.
- S4 produced a marginally higher macro-F1 score, but required significantly more training time and memory.
- S2 Pareto-dominated S3 and S4 when accuracy, epoch time, and GPU memory were considered together.
- S1 remains useful when minimum GPU-memory consumption is the main priority.

The results demonstrate that selectively unfreezing the final ResNet block can provide a better accuracy-efficiency balance than full fine-tuning.

## Project Features

- Automated downloading of the Intel Image Classification dataset
- Configurable ResNet-18 and MobileNetV2 backbones
- Four layer-freezing strategies
- Training and validation accuracy tracking
- Early stopping and learning-rate scheduling
- Test accuracy and macro-F1 evaluation
- GPU-memory and training-time measurement
- Confusion matrices
- Learning curves
- Accuracy-versus-memory and accuracy-versus-time plots
- Automated Pareto-frontier analysis
- Exportable results in CSV, JSON, PNG, and TXT formats

## Repository Structure

```text
Efficient-transfer-learning-s1-s4/
├── README.md
└── efficient_transfer_learning(s1_s4).ipynb
```

## How to Run

1. Open the notebook using the **Open in Google Colab** button above.

2. In Colab, select:

   `Runtime → Change runtime type → T4 GPU`

3. Obtain your Kaggle username and API token from your Kaggle account settings.

4. In Cell 2, replace the credential placeholders with your Kaggle username and token.

   **Never commit or publicly share your actual Kaggle token.**

5. Select:

   `Runtime → Run all`

6. The notebook will automatically:

   - Download and extract the dataset
   - Create training, validation, and testing data loaders
   - Train all four strategies
   - Evaluate accuracy and macro-F1
   - Measure training time and GPU memory
   - Generate comparison plots and confusion matrices
   - Export the final results

## Generated Outputs

After the experiment finishes, the following files are generated:

- `results_summary.csv`
- `full_results.json`
- `learning_curves.png`
- `pareto_plots.png`
- `confusion_matrices.png`
- `metric_bars.png`
- `recommendation.txt`

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Torchinfo
- Kaggle API
- Google Colab

## Authors

- Nandeesh S Mysore Matt
- Charan Arunkumar Raikar
- Rohith K E
- Megha Arakeri

Manipal Institute of Technology Bengaluru  
Manipal Academy of Higher Education, India

## Research Paper

**Efficient Transfer Learning: The Optimal Layer-Freezing Strategies for Accuracy-Time-Memory Tradeoffs in Scene Image Classification**

The study evaluates selective layer unfreezing as a practical approach for obtaining high classification accuracy while reducing training time and GPU-memory consumption.
