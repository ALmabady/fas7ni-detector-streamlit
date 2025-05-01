import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import base64
import io
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import nest_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow FastAPI to run within Streamlit
nest_asyncio.apply()

# ------------------- Model Definition -------------------

class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True
        self.embedding_layer = nn.Linear(self.backbone.embed_dim, embedding_dim)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if features.ndim == 3:
            features = features[:, 0, :]
        embedding = self.embedding_layer(features)
        return F.normalize(embedding, p=2, dim=1)

# ------------------- Load Model & Prototypes -------------------

try:
    model = PrototypicalNetwork(embedding_dim=128)
    model.load_state_dict(torch.load("model_state.pth", map_location="cpu"), strict=False)
    model.eval()
    class_prototypes = torch.load("class_prototypes.pth", map_location="cpu")
except Exception as e:
    logger.error(f"Error loading model or prototypes: {str(e)}")
    st.error("Failed to load model. Please check the model files.")
    raise e

# ------------------- Config -------------------

class_names = [
    'Abu Simbel Temple', 'Bibliotheca Alexandrina', 'Nefertari Temple', 
    'Saint Catherine Monastery', 'Citadel of Saladin', 'Monastery of St. Simeon', 
    'AlAzhar Mosque', 'Fortress of Shali in Siwa', 'Greek Orthodox Cemetery in Alexandria', 
    'Hanging Church', 'khan el khalili', 'Luxor Temple', 'Baron Empain Palace', 
    'New Alamein City', 'Philae Temple', 'Pyramid of Djoser', 'Salt lake at Siwa', 
    'Wadi Al-Hitan', 'White Desert', 'Cairo Opera House', 'Tahrir Square', 
    'Cairo tower', 'Citadel of Qaitbay', 'Egyptian Museum in Tahrir', 
    'Great Pyramids of Giza', 'Hatshepsut temple', 'Meidum pyramid', 
    'Royal Montaza Palace'
]
threshold = 0.35

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------- Prediction Logic -------------------

def predict_image(img):
    try:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img_tensor)
        distances = {
            cls: torch.norm(embedding - proto.to(embedding.device).unsqueeze(0)).item()
            for cls, proto in class_prototypes.items()
        }
        pred_class = min(distances, key=distances.get)
        min_distance = distances[pred_class]
        if min_distance > threshold:
            return "unknown"
        return class_names[pred_class]
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise e

# ------------------- FastAPI Setup -------------------

app = FastAPI(title="Fas7ni Detector API")

class ImageRequest(BaseModel):
    image: str

@app.post("/predict_base64")
async def predict_base64(request: ImageRequest):
    try:
        image_b64 = request.image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        prediction = predict_image(image)
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------- Run FastAPI in a separate thread -------------------

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="info")

fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="Fas7ni Detector", layout="centered")
st.title("Fas7ni Detector 🏛️")
st.write("Upload an image to classify the tourism site.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write("🔍 Classifying...")
        prediction = predict_image(img)
        st.success(f"✅ Predicted Site: **{prediction}**")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        logger.error(f"Streamlit UI error: {str(e)}")
