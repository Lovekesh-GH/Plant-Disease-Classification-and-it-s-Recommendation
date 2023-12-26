
import csv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import io
# from fastai import load_learner
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

# CLASS_NAMES = ['Apple___Apple_scab',
#                 'Apple___Black_rot',
#                 'Apple___Cedar_apple_rust',
#                 'Apple___healthy',
#                 'Blueberry___healthy',
#                 'Cherry_(including_sour)___Powdery_mildew',
#                 'Cherry_(including_sour)___healthy',
#                 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#                 'Corn_(maize)___Common_rust_',
#                 'Corn_(maize)___Northern_Leaf_Blight',
#                 'Corn_(maize)___healthy',
#                 'Grape___Black_rot',
#                 'Grape___Esca_(Black_Measles)',
#                 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                 'Grape___healthy', 
#                 'Orange___Haunglongbing_(Citrus_greening)',
#                 'Peach___Bacterial_spot', 
#                 'Peach___healthy',
#                 'Pepper,_bell___Bacterial_spot',
#                 'Pepper,_bell___healthy', 
#                 'Potato___Early_blight',
#                 'Potato___Late_blight',
#                 'Potato___healthy',
#                 'Raspberry___healthy', 
#                 'Soybean___healthy',
#                 'Squash___Powdery_mildew', 
#                 'Strawberry___Leaf_scorch',
#                 'Strawberry___healthy', 
#                 'Tomato___Bacterial_spot', 
#                 'Tomato___Early_blight',  
#                 'Tomato___Late_blight', 
#                 'Tomato___Leaf_Mold', 
#                 'Tomato___Septoria_leaf_spot', 
#                 'Tomato___Spider_mites Two-spotted_spider_mite', 
#                 'Tomato___Target_Spot', 
#                 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
#                 'Tomato___Tomato_mosaic_virus', 
#                 'Tomato___healthy',
#                 'Background_without_leaves',]

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
                'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
                'Tomato___Tomato_mosaic_virus', 
                'Tomato___healthy',] 


def augment_image(image_bytes):

    try:
        # Open image using PIL
        img = Image.open(io.BytesIO(image_bytes))

        # Example augmentation techniques (you can modify or add more as needed)

        # Rotate the image by 90 degrees
        if img.size != (200, 200):
            img = img.resize((200, 200))
     

        img = img.rotate(90)

        # Flip the image horizontally
        img = ImageOps.mirror(img)

        # Change image contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Increase contrast by a factor of 1.5

        # Convert back to RGB mode if the image isn't in RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Convert the image to a numpy array
        augmented_image = np.array(img)

        return augmented_image

    except Exception as e:
        print(f"Error: {e}")
        return np.zeros((200, 200, 3))


pesticide_mapping = {}

with open('pesticides.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        class_name = row['CLASS_NAMES']
        pesticide_name = row['Pesticide']
        quantity = str(row['Quantity'])

        pesticide_mapping[class_name] = {
            'Pesticide': pesticide_name,
            'Quantity': quantity
        }


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


# @app.post("/predict")
# async def predict(
#     file: UploadFile = File(...)
# ):
#     # ... (your existing code for model prediction)
#     img = await file.read()
#     image = read_file_as_image(img)
#     # augmented_img_array = augment_image(io.BytesIO(image))
#     augmented_img_array = augment_image(image)
#     img_batch = np.expand_dims(augmented_img_array, 0)

#     prediction = MODEL.predict(img_batch)
#     predicted_class = CLASS_NAMES[np.argmax(prediction)]
#     confidence = np.max(prediction)

#     # Check if the predicted class is in the pesticide mapping
#     if predicted_class in pesticide_mapping:
#         pesticide_info = pesticide_mapping[predicted_class]
#         return {
#             "predicted_class": predicted_class,
#             "Pesticide": pesticide_info['Pesticide'],
#             "Quantity": pesticide_info['Quantity'],
#             "confidence": float(confidence),
#         }
#     else:
#         return {
#             "predicted_class": predicted_class,
#             "error": "Pesticide information not found for this class",
#             "confidence": float(confidence),
#         }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        img = await file.read()
        # image_data = read_file_as_image(img)
        augmented_img_array = augment_image(img)
        
        img_batch = np.expand_dims(augmented_img_array, 0)

        prediction = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Check if the predicted class is in the pesticide mapping
        if predicted_class in pesticide_mapping:
            pesticide_info = pesticide_mapping[predicted_class]
            return {
                "predicted_class": predicted_class,
                "Pesticide": pesticide_info['Pesticide'],
                "Quantity": pesticide_info['Quantity'],
                "confidence": float(confidence),
            }
        # else:
        #     return {
        #         "predicted_class": predicted_class,
        #         "error": "Pesticide information not found for this class",
        #         "confidence": float(confidence),
        #     }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "error": "Failed to process the image",
        }
    
if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)