import os
import sys
import cv2
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

sys.path.append("./src/")

try:
    from ViT import ViTWithClassifier
    from utils import device_init, config_files
    from prompt import classifier_prompt, QA_prompt

except ImportError:
    print("Error: Unable to import required modules.".capitalize())

class MedicalAssistant:
    def __init__(self, device: str = "cuda", image: str = None):
        self.device = device
        self.image = image

        self.image_channels = 1

        self.device = device_init(device=device)

        self.classifier = ViTWithClassifier(
            image_channels=config_files()["dataloader"]["image_channels"],
            image_size=config_files()["dataloader"]["image_size"],
            patch_size=config_files()["ViT"]["patch_size"],
            target_size=config_files()["dataloader"]["target_size"],
            encoder_layer=config_files()["ViT"]["encoder_layer"],
            nhead=config_files()["ViT"]["nhead"],
            d_model=config_files()["ViT"]["d_model"],
            dim_feedforward=config_files()["ViT"]["dim_feedforward"],
            dropout=config_files()["ViT"]["dropout"],
            activation=config_files()["ViT"]["activation"],
            layer_norm_eps=float(config_files()["ViT"]["layer_norm_eps"]),
            bias=config_files()["ViT"]["bias"],
        ).to(self.device)

        self.memory = []

    def load_model(self):
        path = "./artifacts/checkpoints/best_model/best_model.pth"
        model = torch.load(path)
        state_dict = model["model_state_dict"]
        self.classifier.load_state_dict(state_dict=state_dict)

    def preprocess_image(self):
        if self.image_channels == 1:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.CenterCrop((224, 224)),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

        image = cv2.imread(self.image)
        image = Image.fromarray(image)
        image = transform(image)
        image = image.unsqueeze(0)

        return image.to(self.device)

    def chat(self):
        image = self.preprocess_image()
        self.classifier.eval()

        with torch.no_grad():
            predicted = self.classifier(image)
            score = torch.softmax(input=predicted, dim=1)
            score, _ = torch.max(score, dim=1)
            score = f"{round(score.item() * 100, 2)}%"

            predicted = torch.argmax(predicted, dim=1)[0]

            if predicted == 0:
                labels = "Brain: Glioma"
            elif predicted == 1:
                labels = "Brain: Meningioma"
            elif predicted == 2:
                labels = "Brain: No Tumor"
            else:
                labels = "Brain: Pituitary"

        load_dotenv()
        llm = ChatOpenAI(model = gpt-4.1)
        parser = StrOutputParser()

        initial_response = classifier_prompt | llm | parser
        initial_response = initial_response.invoke(
            {"predicted_disease": labels, "predicted_probability": score}
        )

        st.write("### AI Initial Response")
        st.write(initial_response)

        self.memory.append("AI Response:\n" + initial_response)

        st.write("### Ask Follow-up Questions")
        for i in range(5):  # Allow up to 5 questions
            question = st.text_input(f"Your Question {i+1}:", key=f"q{i}")
            if question:
                full_prompt = "\n".join(self.memory) + "\nHuman: " + question
                response = QA_prompt | llm | parser
                answer = response.invoke({"question": full_prompt})
                self.memory.append("AI Response:\n" + answer)
                st.write(f"**AI Response {i+1}:**")
                st.write(answer)


st.title("Medical Assistant Chatbot")

uploaded_image = st.file_uploader("Upload a brain MRI scan image", type=["jpg", "jpeg", "png"])
device = "cuda" if torch.cuda.is_available() else "cpu"

if uploaded_image is not None:
    image_path = "./temp_uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.read())
        
    assistant = MedicalAssistant(device=device, image=image_path)
    
    assistant.load_model()
    assistant.chat()
