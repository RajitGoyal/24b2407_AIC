# Q2. Transfer Learning for Fashion-MNIST
## Overview
This project demonstrates transfer learning for image classification on the Fashion-MNIST dataset. A pretrained CNN (ResNet50) is adapted to classify 28×28 grayscale fashion images into 10 classes, following the AI Community Assignment 2025 guidelines.

## Task Objective
### Adapt a pretrained CNN (ResNet50) to classify Fashion-MNIST images

### Implement a data pipeline:

Resize images to 224×224

Convert grayscale (1 channel) to RGB (3 channels)

### Model setup:

Load pretrained backbone without top layers

Freeze backbone and add a new fully connected head

Train only the head, record validation metrics

### Fine-tuning:

Unfreeze selected deeper blocks

Fine-tune with a lower learning rate

Experiment with data augmentation, learning rate scheduling, dropout, and weight decay

## Data
Dataset: Fashion-MNIST

Access: Automatically loaded using TensorFlow/Keras

## Approach
### 1.Data Pipeline

Loaded Fashion-MNIST using Keras

Resized images from 28×28 to 224×224

Converted grayscale images to 3-channel RGB by duplication

Applied data augmentation (random flips, rotations, zooms) for better generalization

### 2.Model Construction

Used ResNet50 pretrained on ImageNet, excluding the top classification layers

Backbone was frozen initially to preserve pretrained features

Added a new classification head:
GlobalAveragePooling → Dense(256, relu, L2 regularization) → Dropout → Dense(10, softmax)

### 3.Training

Trained only the new head for 10 epochs using Adam optimizer with a scheduled learning rate

Used mixed precision for faster training on GPU

### 4.Fine-Tuning

Unfroze the last 16 layers of ResNet50

Continued training with a lower learning rate (1e-5) for 5 epochs

Further improved accuracy and adapted model to Fashion-MNIST

## Experimentation

Data augmentation: Improved robustness to input variations

Learning rate scheduling: Used exponential decay for smoother convergence

Dropout & L2 regularization: Prevented overfitting

## Evaluation

Evaluated final model on the test set

Plotted validation accuracy and loss curves for both training phases

## Files
Q2_FashionMNIST_TransferLearning.ipynb — Main notebook (code + explanations)




# Q3: Retrieval-Augmented Generation (RAG) Chatbot over PDF
## Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot over a PDF document using open-source LLMs, as required by the AI Community Assignment 2025. The chatbot can answer questions about the PDF by retrieving relevant document sections and generating grounded answers.
Bonus features include:

Multi-turn conversational memory (history-aware responses)

Knowledge graph extraction and visualization

KV-cache for repeated queries

Modular, agentic architecture

## Features
### PDF Ingestion & Chunking:
Parses the PDF and splits it into semantic text chunks for efficient retrieval.

### Vector Indexing:
Uses FAISS and sentence-transformers to embed and index chunks for semantic search.

### Retrieval & Generation:
For each user query, retrieves the most relevant chunks and generates an answer using the Groq API with Llama 3.

### Bonus Techniques:

KV-cache: Caches answers for repeated queries to speed up responses.

Knowledge Graph: Extracts simple entities and relations, building a mini knowledge graph and visualizing it.

History-Aware Responses: Maintains chat memory so the bot can use previous questions and answers for better multi-turn conversation.

Agentic Architecture: Modular agents for information extraction, synthesis, and query answering, coordinated by a central workflow manager.

## File Structure
AICommunity_Assignment_25.ipynb — The main notebook with all code, explanations, and output.

AICommunity_Assignment_25.pdf — The PDF document used for RAG.

## How to Run
Upload the notebook and PDF to Google Colab.

Install dependencies (see the first code cell).

Set your Groq API key in the designated cell.

Run all cells in order.

Interact with the chatbot in the main chat loop cell.

The knowledge graph will be displayed automatically.
