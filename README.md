# GraphKWS: Graph Neural Networks for Keyword Spotting  

[![Paper](https://img.shields.io/badge/paper-PDF-blue)](./GraphKWS.pdf)  

This repository contains the code accompanying the paper:  
**GraphKWS: Exploring Graph Neural Networks for Keyword Spotting**  
*Eleonora Mesaglio, Marlon Joshua Helbing*  

---

## 📖 Overview  

Keyword Spotting (KWS) enables devices to detect specific spoken words efficiently, a critical task for voice assistants and embedded systems.  

In this work, we propose **Graph Neural Network (GNN)** architectures for KWS, representing audio spectrograms as graphs where temporal frames are modeled as nodes. Our contributions include:  

- **Graph-based representation of audio** using sparse, cosine similarity–based adjacency with dilated connectivity.  
- **Balanced message passing** via Graph Convolutional layers and attention mechanisms for noise robustness.  
- **Reduced node representations** for flexible trade-offs between accuracy and efficiency.  

Our models achieve:  
- **Low-footprint variant (18.4k parameters)**: matches state-of-the-art accuracy while reducing multiplies by ×3.  
- **Medium-footprint variant (151k parameters)**: reaches **94.68% accuracy**, using **23% fewer parameters** than previous attention-based approaches.  

To our knowledge, this is the **first work applying GNNs to keyword spotting**, demonstrating their potential as lightweight and scalable alternatives to CNN-based models.  

---

## 🚀 Features  

- Graph construction from audio frames with **cosine similarity** and **dilated adjacency**.  
- Support for **MFCC** and **GFCC** feature extraction.  
- **SpecAugment** and **time-shift** data augmentation.  
- Multiple GNN variants:  
  - **Base GNN** (+ ablation study on finding optimal hyperparameters)  
  - **GCN** (low-footprint)  
  - **GAT-GCN** (attention-based, medium-footprint)  
- Benchmarking on the **Google Speech Commands Dataset V2**.  

---

## 📈 Results  

| Model                | Params | Multiplies | Accuracy (%) |
|----------------------|--------|------------|--------------|
| res8-narrow (CNN)    | 20.8k  | 5.65M      | 85.93        |
| **GCN (ours)**       | 18.4k  | 1.75M      | 87.81        |
| CRNN + attention     | 202k   | –          | 93.9         |
| **GAT-GCN (ours)**   | 151k   | 11.1M      | **94.68**    |

---
