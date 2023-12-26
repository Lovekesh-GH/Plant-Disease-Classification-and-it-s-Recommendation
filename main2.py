from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
# import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("./saved_models/model.h5")

# CLASS_NAMES = ['Pepper__bell___Bacterial_spot',
#  'Pepper__bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Tomato_Bacterial_spot',
#  'Tomato_Early_blight',
#  'Tomato_Late_blight',
#  'Tomato_Leaf_Mold',
#  'Tomato_Septoria_leaf_spot',
#  'Tomato_Spider_mites_Two_spotted_spider_mite',
#  'Tomato__Target_Spot',
#  'Tomato__YellowLeaf__Curl_Virus',
#  'Tomato__mosaic_virus',
#  'Tomato_healthy'] 3

CLASS_NAMES = ['Apple___Apple_scab',
                'Apple___Black_rot',
                'Apple___Cedar_apple_rust',
                'Apple___healthy',
                'Blueberry___healthy',
                'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy',
                'Grape___Black_rot',
                'Grape___Esca_(Black_Measles)',
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 
                'Orange___Haunglongbing_(Citrus_greening)',
                'Peach___Bacterial_spot', 
                'Peach___healthy',
                'Pepper,_bell___Bacterial_spot',
                'Pepper,_bell___healthy', 
                'Potato___Early_blight',
                'Potato___Late_blight', 
                'Potato___healthy',
                'Raspberry___healthy', 
                'Soybean___healthy',
                'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch',
                'Strawberry___healthy', 
                'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 
                'Tomato___Late_blight', 
                'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 
                'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato___Tomato_mosaic_virus', 
                'Tomato___healthy']

# CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"] 1
# CLASS_NAMES = [
#     "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy",
#     "Blueberry Healthy", "Cherry Powdery Mildew", "Cherry Healthy",
#     "Corn (Maize) Cercospora Gray Leaf Spot", "Corn (Maize) Common Rust",
#     "Corn (Maize) Northern Leaf Blight", "Corn (Maize) Healthy",
#     "Grape Black Rot", "Grape Esca (Black Measles)",
#     "Grape Leaf Blight (Isariopsis Leaf Spot)", "Grape Healthy",
#     "Orange Haunglongbing (Citrus Greening)", "Peach Bacterial Spot",
#     "Peach Healthy", "Pepper Bell Bacterial Spot", "Pepper Bell Healthy",
#     "Potato Early Blight", "Potato Late Blight", "Potato Healthy",
#     "Raspberry Healthy", "Soybean Healthy", "Squash Powdery Mildew",
#     "Strawberry Leaf Scorch", "Strawberry Healthy", "Tomato Bacterial Spot",
#     "Tomato Early Blight"
# ] 


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

    prediction = MODEL.predict(img_batch)
    

    # json_data = {
    #     "instances": img_batch.tolist()
    # }

    # response = requests.post(endpoint, json=json_data)
    # prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    print(predicted_class)
    print(confidence)

    return {
        "class": predicted_class,
        "confidence": float(confidence),
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)