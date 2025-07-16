from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# CORS Middleware to allow React frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model (only once during server startup)
try:
    model = tf.keras.models.load_model('models/final_acne_model.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Image preparation function
def prepare_image(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))  # Resize to match model input size
        img_array = np.array(img) / 255.0  # Normalize the image
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error in preparing image: {e}")
        return None

# Prediction function
def predict_image(image_bytes, filename="uploaded_image"):
    if model is None:
        return {"error": "Model not loaded."}
    image = prepare_image(image_bytes)
    if image is not None:
        prediction = model.predict(image)
        print(f"Prediction for {filename}: {prediction}")
        severity_probabilities = prediction[0]
        predicted_class_index = np.argmax(severity_probabilities)
        confidence = severity_probabilities[predicted_class_index]

        # Assuming the order of classes during training was:
        # 0: clear_skin, 1: mild_acne, 2: severe_acne
        if predicted_class_index == 0:
            acne_severity = "Clear Skin"
            recommendation = "Keep your skin healthy by washing gently, moisturizing daily, and always using sunscreen!"
        elif predicted_class_index == 1:
            acne_severity = "Mild Acne"
            recommendation = "Try acne washes with benzoyl peroxide (2.5-5%) and spot treatments with salicylic acid."
        elif predicted_class_index == 2:
            acne_severity = "Severe Acne"
            recommendation = "See a skin doctor for strong treatments—don’t squeeze pimples to avoid scars!"
        else:
            acne_severity = "Unknown"
            recommendation = "Could not determine severity."

        return {"acne_severity": acne_severity, "recommendation": recommendation, "confidence": f"{confidence:.4f}"}
    else:
        return {"error": "Error in image processing."}

# Define the /predict route to handle image uploads
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        prediction = predict_image(contents, image.filename)
        return JSONResponse(content=prediction)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Run the app (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)