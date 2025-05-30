import os
import sys
import cv2
import math
import torch
import torch.nn as nn
from PIL import Image
import argparse
from torchvision import transforms
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

sys.path.append("./src/")

from utils import device_init
from ViT import ViTWithClassifier
from prompt import classifier_prompt, QA_prompt


class MedicalAssistant:
    def __init__(self, device: str = "cuda", image: str = None):
        self.device = device
        self.image = image

        self.image_channels = 1

        self.device = device_init(device=device)

        self.classifier = ViTWithClassifier(
            image_channels=1,
            image_size=224,
            patch_size=16,
            target_size=4,
            encoder_layer=1,
            nhead=8,
            d_model=256,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            layer_norm_eps=1e-05,
            bias=False,
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
                labels = "Brain: glioma".title()
            elif predicted == 1:
                labels = "Brain: Meningioma".title()
            elif predicted == 2:
                labels = "Brain: No Tumor".title()
            else:
                labels = "Brain: Pituitary".title()

        load_dotenv()

        llm = ChatOpenAI()
        parser = StrOutputParser()

        initial_response = classifier_prompt | llm | parser
        initial_response = initial_response.invoke(
            {"predicted_disease": labels, "predicted_probability": score}
        )

        print(initial_response)

        self.memory.append("AI Response:\n" + " " + initial_response)

        while True:
            question = input("Human: ")
            if question == "exit":
                break

            question = "\n".join(self.memory) + "\nHuman: " + question

            response = QA_prompt | llm | parser
            response = response.invoke({"question": question})

            self.memory.append("AI Response:\n" + " " + response)

            print("AI:\n", response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Medical Assistant chatbot".title())
    parser.add_argument("--image", type=str, help="Path to the image file".capitalize())
    parser.add_argument(
        "--device", type=str, help="Device to run the model on".capitalize()
    )

    args = parser.parse_args()

    assistant = MedicalAssistant(device=args.device, image=args.image)

    assistant.load_model()
    assistant.chat()
