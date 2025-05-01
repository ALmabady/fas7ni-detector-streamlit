import streamlit as st
from fastapi import FastAPI, UploadFile
from starlette.middleware.wsgi import WSGIMiddleware
from PIL import Image
import io
import torch
from torchvision import transforms
import timm
import torch.nn as nn
import torch.nn.functional as F

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
model = PrototypicalNetwork(embedding_dim=128)
state_dict = torch.load("model_state.pth", map_location=torch.device("cpu"), weights_only=True)
model.load_state_dict(state_dict)
class_prototypes = torch.load("class_prototypes.pth", map_location=torch.device("cpu"))

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
        model.eval()
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
        img = Image.open(io.BytesIO(contents))
        prediction = predict_image(img)
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}

# ----- Streamlit UI -----
st.title("Fas7ni Detector")
st.write("Upload an image to identify the tourism site.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    prediction = predict_image(img)
    st.write(f"Name of site is: {prediction}")

# Mount FastAPI app (runs on same port as Streamlit)
app = st.server.server.app
app.mount("/api", WSGIMiddleware(fastapi_app))