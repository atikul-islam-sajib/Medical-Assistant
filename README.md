# **🧠 Medical Assistant - AI-Powered Brain Tumor Classifier & Chatbot**

This is a **fully customizable, transformer-based** medical diagnostic system designed to classify brain tumors from MRI scans and assist with interactive medical Q&A using **LLMs**.

> **"AI-Powered Brain Tumor Detection + Chatbot using Vision Transformers (ViT) & LangChain"**  
> Built by deep learning enthusiasts, this unofficial project explores the combination of **Vision Transformers (ViT)** and **LLMs (OpenAI)** for medical assistance.

---

## 🖼️ Demo Screenshots

<div align="center">
  <img src="./logs/initial.png" alt="Initial Diagnosis" width="48%" style="margin-right: 2%;" />
  <img src="./logs/initial2.png" alt="Interactive Chat Q&A" width="48%" />
</div>

---

## **📌 Key Features**
✅ ViT-based image classifier for brain MRI scans  
✅ Multi-class classification: **Glioma, Meningioma, Pituitary, No Tumor**  
✅ Integrated **LangChain + OpenAI ChatGPT** for contextual medical Q&A  
✅ Support for **custom model training** with multiple optimizers & regularization  
✅ Automatic **image preprocessing** and device-aware pipeline  
✅ Saves **model checkpoints** and tracks **training performance**  

---

## **📌 Model Architecture**

The **Medical Assistant** is composed of two primary modules:

### 🧠 1. ViT Classifier
A **Vision Transformer (ViT)** with a custom classifier head:
- Patch-based encoding (16x16)
- Transformer encoder layers
- Fully connected classifier

### 💬 2. Chatbot Interface
An **LLM-powered medical assistant**:
- Uses predictions to trigger AI responses
- Supports dynamic follow-up Q&A
- Powered by LangChain & OpenAI's `gpt-3.5-turbo`

```
  Image
   │
   ▼
🧠 ViT Classifier ─────────┐
   │                      │
   ▼                      ▼
Tumor Class + Score   ─▶  LangChain + LLM Chatbot
                         (Interactive Medical Q&A)
```

---

## **📌 Installation**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/medical-assistant.git
cd medical-assistant
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Setup OpenAI API**
Create a `.env` file:
```env
OPENAI_API_KEY=your-openai-key
```

---

### 📌 **Configuration Parameters**

| **Parameter**        | **Description**                               |
|----------------------|-----------------------------------------------|
| `--image_path`       | Path to the dataset                           |
| `--image_channels`   | Number of image channels                      |
| `--image_size`       | Image size                                    |
| `--batch_size`       | Batch size                                    |
| `--split_size`       | Split size for train/test                     |
| `--epochs`           | Number of training epochs                     |
| `--lr`               | Learning rate                                 |
| `--beta1`            | Beta1 value for Adam optimizer                |
| `--beta2`            | Beta2 value for Adam optimizer                |
| `--weight_decay`     | Weight decay for regularization               |
| `--momentum`         | Momentum for SGD optimizer                    |
| `--adam`             | Use Adam optimizer                            |
| `--SGD`              | Use SGD optimizer                             |
| `--optimizer`        | Optimizer to use: `adam` or `sgd` (recommended)|
| `--device`           | Device to use: `cuda`, `cpu`, or `mps`        |
| `--verbose`          | Show logs, plots, save outputs                |
| `--dataset`          | Dataset path used for testing                 |
| `--image`            | Path to MRI image for inference               |
| `--train`            | Train the model                               |
| `--test`             | Test the model                                |

---

## **📌 Running the Assistant**

### 🧪 Inference + Chatbot

```bash
python medical_assistant.py --image ./sample.jpg --device cuda
```

💬 Type `exit` to stop the chatbot.

---

## **📌 Model Training**

```bash
python train.py --epochs 50 --lr 0.0003 --adam True --device cuda
```

Supports:
- Adam / SGD optimizers
- L1 & ElasticNet regularization
- Checkpoint saving & accuracy metrics

---

## **📌 Output Structure**

| **Process**         | **Path**                                             |
|---------------------|------------------------------------------------------|
| Best Model          | `./artifacts/checkpoints/best_model/best_model.pth` |
| Training Checkpoints| `./artifacts/checkpoints/train_models/`             |
| Input Images        | Supplied via `--image` path or folder               |
| Dialogue Memory     | Stored internally during a session                  |

---

## **📌 Workflow**

| **Step** | **Process**        | **Description** |
|----------|--------------------|-----------------|
| 1️⃣       | Preprocess Image   | Resize, grayscale, normalize |
| 2️⃣       | Classify Image     | Predict tumor class using ViT |
| 3️⃣       | Display Diagnosis  | Output tumor type & confidence |
| 4️⃣       | Chat with AI       | Ask related medical questions |

---

## **📌 Model Classes: Brain Tumor**

| Label Index | Prediction       |
|-------------|------------------|
| `0`         | Brain: Glioma     |
| `1`         | Brain: Meningioma |
| `2`         | Brain: No Tumor   |
| `3`         | Brain: Pituitary  |

---

## **📌 License**

This project is released under the **MIT License**.  
**Disclaimer:** This is an experimental project and not for clinical use.