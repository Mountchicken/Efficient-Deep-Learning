# Efficient Deep Learning

<div align=center>
  <img src='images/cover.png' width=180 >
</div>
<div align=center>
  <p ><strong>Tricks here, you can have</strong></p>
</div>

## 1.Introduction

- With the rapid development of deep learning, more and more people are flocking to this field, including me. As a former rookie, I have experienced various problems during my deep learning process, so I create this repo here to record some tricks that can make you have an efficient deep learning. You are also welcome to raise a PR and give some of your tips!

## 2.Efficient Coding

- Strategies to code efficiently.
- [Efficient Coding](Efficient_Coding.md)
  - [Use Vscode](Efficient_Coding.md#1-you-shouldnt-miss-vscode)
  - [Auto code formating](Efficient_Coding.md#2-automatically-format-your-code)
  - [Pre-commit hook](Efficient_Coding.md#3-use-a-pre-commit-hook-to-check-your-code)
  - [Learn to use git](Efficient_Coding.md#4-learn-to-use-git)
  - [Grammarly](Efficient_Coding.md#5-use-grammarly-to-check-your-writing)
  - [StackOverflow](Efficient_Coding.md#6-search-on-stackoverflow-first)
  - [Auto docstring](Efficient_Coding.md#7-automatically-format-your-docstring)
## 3.Efficient Data Processing

- Strategies to speed up your data processing.
- [Efficient Data Processing](Efficient_DataProcessing.md)
  - [SSD](Efficient_DataProcessing.md#11-use-ssd-instead)
  - [num_workers and pin_memory](Efficient_DataProcessing.md#12-multiple-workers-and-pinmemory-in-dataloader)
  - [LMDB file](Efficient_DataProcessing.md#21-efficient-data-storage-methods)
  - [Albumentations](Efficient_DataProcessing.md#22-efficient-data-augmentation-library)
  - [Data augmentation on GPU](Efficient_DataProcessing.md#23-data-augmentation-on-gpu)

## 4.Efficient Training

- Strategies to speed up your training process.
- [Efficient Traininig](Efficient_Training.md)
  - [cudnn.benchmark=True](Efficient_Training.md#11-set-cudnnbenchmarktrue)
  - [Set gradients to None during back propagation](Efficient_Training.md#12-set-gradients-to-none-during-back-propagation)
  - [Turn off debugging APIs](Efficient_Training.md#13-turn-off-debugging)
  - [Turn off gradient computation during validation](Efficient_Training.md#14-turn-off-gradient-computation-during-validation)
  - [Use another optimizer AdamW](Efficient_Training.md#21-use-another-optimizer-adamw)
  - [Learning rate schedule](Efficient_Training.md#22-learning-rate-schedule)
  - [Useful combination, Adam with 3e-4](Efficient_Training.md#23-best-combination-adam-with-3e-4)

## 5.Efficient GPUtilization

- Strategies to have a better GPU utilization.
- [Efficient GPUtilization](Efficient_GPUtilization.md)
  - [CUDA out of memory solutions](Efficient_GPUtilization.md#1-cuda-out-of-memory-solutions)
  - [Automatic Mixed Precision (AMP)](Efficient_GPUtilization.md#21-automatic-mixed-precisionamp)
  - [Gradient Accumulation](Efficient_GPUtilization.md#22-gradient-accumulation)
  - [Gradient Checkpoint](Efficient_GPUtilization.md#23-gradient-checkpoint)
  - [Data parallelization training](Efficient_GPUtilization.md#31-distributed-model-training)

## 6.Efficient Tools

- A list of useful tools.
- [Efficient Tools](Efficient_Tools.md)
  - [Torchinfo: Visualize Network Architecture](Efficient_Tools.md#1-torchinfo-visualize-network-architecture)
  - [drawio: Free graphing software](Efficient_Tools.md#2-drawio-free-graphing-software)
  - [Octotree: Free gitHub code tree](Efficient_Tools.md#3-octotree-free-github-code-tree)
  - [ACRONYMIFY: Name your paper with a cool acronyms](Efficient_Tools.md#4-acronymify-name-your-paper-with-a-cool-acronyms)
  - [Linggle: Grammer checker](Efficient_Tools.md#5-linggle-grammer-checker)
----

- ***"The past decade has seen tremendous progress in the field of artificial intelligence thanks to the resurgence of neural networks through deep learning. This has helped improve the ability for computers to see, hear, and understand the world around them, leading to dramatic advances in the application of AI to many fields of science and other areas of human endeavor" ——Jeffrey Dean***