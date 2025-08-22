import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib

# ----------------------------
# Load model
# ----------------------------
@st.cache_resource
def load_model():
    # Load CPU-safe checkpoint
    model_data = joblib.load("crop_classifier_model_cpu.pkl")

    # Define model
    model = models.resnet18(pretrained=False)
    num_classes = len(model_data["class_to_idx"])
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Load weights
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()
    return model, {v: k for k, v in model_data["class_to_idx"].items()}  # invert mapping


model, idx_to_class = load_model()

# ----------------------------
# Image Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])])

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌱 Crop Classifier App")
st.write("Upload an image of a crop plant leaf to predict its class.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        top_prob, top_class = torch.max(probs, 1)

    pred_class = idx_to_class[top_class.item()]
    confidence = top_prob.item() * 100

    st.success(f"🌿 Prediction: **{pred_class}** ({confidence:.2f}% confidence)")
