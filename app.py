import streamlit as st
from fastapi import FastAPI, UploadFile, Body
from fastapi.middleware.wsgi import WSGIMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms
import timm
import torch.nn as nn
import torch.nn.functional as F
import base64
from starlette.applications import Starlette
from starlette.routing import Mount
from streamlit.web.server import Server

# ----- Model Architecture -----
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(PrototypicalNetwork, self).__init__()
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, 'blocks'):
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

# ----- Load Model and Prototypes -----
try:
    model = PrototypicalNetwork(embedding_dim=128)
    state_dict = torch.load("model_state.pth", map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    class_prototypes = torch.load("class_prototypes.pth", map_location=torch.device("cpu"))
except Exception as e:
    st.error(f"Failed to load model or prototypes: {str(e)}")
    raise

# ----- Image Preprocessing Transform -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ----- Class Names and Threshold -----
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

# ----- Prediction Function -----
def predict_image(img):
    try:
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embedding = model(img_tensor)
        distances = {cls: torch.norm(embedding - proto.to(embedding.device).unsqueeze(0)).item() 
                     for cls, proto in class_prototypes.items()}
        pred_class = min(distances, key=distances.get)
        min_distance = distances[pred_class]
        if min_distance > threshold:
            return "unknown"
        return class_names[pred_class]
    except Exception as e:
        return f"Error: {str(e)}"

# ----- FastAPI Setup -----
fastapi_app = FastAPI()

@fastapi_app.post("/predict")
async def predict(file: UploadFile):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = predict_image(img)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

@fastapi_app.post("/predict_base64")
async def predict_base64(data: dict = Body(...)):
    try:
        base64_string = data["image"]
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        prediction = predict_image(img)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)} 

# ----- Streamlit UI -----
def streamlit_app():
    st.title("Fas7ni Detector")
    st.write("Upload an image to identify the tourism site.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.write("Classifying...")
            prediction = predict_image(img)
            st.write(f"Name of site is: {prediction}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ----- Combine FastAPI and Streamlit -----
app = Starlette(routes=[
    Mount("/api", app=fastapi_app),
    Mount("/", app=WSGIMiddleware(st.web.cli_main))
])

# Run Streamlit in the main app
if __name__ == "__main__":
    streamlit_app()
