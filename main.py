from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update the endpoint to your specific model's API endpoint
# endpoint = "http://localhost:8501/v1/models/your_model_name:predict"
endpoint = "/home/lovekesh/Desktop/Project_Pesticides/checkpoints"

# # Define your class names based on the provided diseases
CLASS_NAMES = [
    "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy",
    "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
    "Corn (Maize) Cercospora Gray Leaf Spot", "Corn (Maize) Common Rust",
    "Corn (Maize) Northern Leaf Blight", "Corn (Maize) Healthy",
    "Grape Black Rot", "Grape Esca (Black Measles)",
    "Grape Leaf Blight (Isariopsis Leaf Spot)", "Grape Healthy",
    "Orange Haunglongbing (Citrus Greening)", "Peach Bacterial Spot",
    "Peach Healthy", "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
    "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
    "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
    "Strawberry Leaf Scorch", "Strawberry Healthy", "Tomato Bacterial Spot",
    "Tomato Early Blight"
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    return {
        "class": predicted_class,
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)