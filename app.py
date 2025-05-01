# Import Streamlit for the web UI
import streamlit as st

# Import FastAPI and related modules for the API
from fastapi import FastAPI, UploadFile, Body

# Import FastAPI middleware to integrate with Streamlit
from fastapi.middleware.wsgi import WSGIMiddleware

# Import PIL for image processing
from PIL import Image

# Import io for handling byte streams
import io

# Import PyTorch and related libraries for the model
import torch
from torchvision import transforms
import timm

# Import PyTorch neural network module
import torch.nn as nn

# Import functional module for normalization
import torch.nn.functional as F

# Import base64 for decoding base64-encoded images
import base64

# Import uvicorn to run the combined server
import uvicorn

# Import Starlette for mounting routes
from starlette.applications import Starlette
from starlette.routing import Mount

# Define the PrototypicalNetwork class for the model
class PrototypicalNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(PrototypicalNetwork, self).__init__()
        # Use DeiT-small without pretrained weights
        self.backbone = timm.create_model('deit_small_patch16_224', pretrained=False)
        # Freeze backbone parameters to prevent training
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Unfreeze the last two blocks for fine-tuning
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True
        # Add a linear layer to project to embedding dimension
        self.embedding_layer = nn.Linear(self.backbone.embed_dim, embedding_dim)
    
    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone.forward_features(x)
        # Handle feature dimensions (remove cls token if needed)
        if features.ndim == 3:
            features = features[:, 0, :]
        # Project to embedding space
        embedding = self.embedding_layer(features)
        # Normalize the embedding
        return F.normalize(embedding, p=2, dim=1)

# Load the model and class prototypes with error handling
try:
    # Initialize the model with 128-dimensional embeddings
    model = PrototypicalNetwork(embedding_dim=128)
    # Load the model state dictionary from GitHub repo
    state_dict = torch.load("model_state.pth", map_location=torch.device("cpu"), weights_only=True)
    # Apply the state dictionary to the model
    model.load_state_dict(state_dict)
    # Set the model to evaluation mode
    model.eval()
    # Load class prototypes from GitHub repo
    class_prototypes = torch.load("class_prototypes.pth", map_location=torch.device("cpu"))
except Exception as e:
    # Display error in Streamlit UI
    st.error(f"Failed to load model or prototypes: {str(e)}")
    # Raise the exception to halt execution
    raise

# Define image preprocessing transforms
transform = transforms.Compose([
    # Resize to 224x224 as expected by the model
    transforms.Resize((224, 224)),
    # Convert PIL image to tensor
    transforms.ToTensor(),
    # Normalize with ImageNet mean and std
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names for predictions
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

# Define the distance threshold for "unknown" class
threshold = 0.35

# Define the prediction function
def predict_image(img):
    try:
        # Apply transforms and add batch dimension
        img_tensor = transform(img).unsqueeze(0)
        # Disable gradient computation for inference
        with torch.no_grad():
            # Get the embedding from the model
            embedding = model(img_tensor)
        # Calculate distances to class prototypes
        distances = {cls: torch.norm(embedding - proto.to(embedding.device).unsqueeze(0)).item() 
                     for cls, proto in class_prototypes.items()}
        # Find the class with the minimum distance
        pred_class = min(distances, key=distances.get)
        # Get the minimum distance
        min_distance = distances[pred_class]
        # Return "unknown" if distance exceeds threshold
        if min_distance > threshold:
            return "unknown"
        # Return the predicted class name
        return class_names[pred_class]
    except Exception as e:
        # Return error message as string
        return f"Error: {str(e)}"

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Define the /predict endpoint for file uploads
@fastapi_app.post("/predict")
async def predict(file: UploadFile):
    try:
        # Read the uploaded file contents
        contents = await file.read()
        # Open the image and convert to RGB
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        # Get the prediction
        prediction = predict_image(img)
        # Return JSON response with prediction
        return {"prediction": prediction}
    except Exception as e:
        # Return JSON error response with 400 status
        return {"prediction": f"Error: {str(e)}"}, 400

# Define the /predict_base64 endpoint for base64-encoded images
@fastapi_app.post("/predict_base64")
async def predict_base64(data: dict = Body(...)):
    try:
        # Extract the base64 string from the request
        base64_string = data["image"]
        # Decode the base64 string
        img_data = base64.b64decode(base64_string)
        # Open the image and convert to RGB
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        # Get the prediction
        prediction = predict_image(img)
        # Return JSON response with prediction
        return {"prediction": prediction}
    except Exception as e:
        # Return JSON error response with 400 status
        return {"prediction": f"Error: {str(e)}"}, 400

# Define a simple Streamlit WSGI app
def streamlit_app(environ, start_response):
    # Run the Streamlit app
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
    # Return a dummy response (Streamlit handles rendering)
    status = '200 OK'
    headers = [('Content-type', 'text/html')]
    start_response(status, headers)
    return [b"Streamlit app"]

# Create a Starlette app to mount FastAPI and Streamlit
app = Starlette(routes=[
    # Mount FastAPI at /api
    Mount("/api", app=fastapi_app),
    # Mount Streamlit at /
    Mount("/", app=WSGIMiddleware(streamlit_app))
])

# Run the combined app with uvicorn if executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="error")
