from fastapi import FastAPI, File, UploadFile, Header, HTTPException
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Render Environment Variable'dan API anahtarını oku
API_KEY = os.environ.get("YOLO_API_KEY", "varsayilan_anahtar")

# ONNX modelini yükle (CPU üzerinde çalışacak)
try:
    session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print("✅ ONNX modeli başarıyla yüklendi.")
except Exception as e:
    print(f"❌ Model yüklenirken hata: {e}")
    session = None

# Sınıf isimleri (Eğitimdeki sırayla aynı olmalı)
CLASS_NAMES = [
    "Üçgen", "Çift Tepe", "Çift Dip", "Yükselen Kanal", "Düşen Kanal",
    "OBO", "TOBO", "Çanak", "Çanak-Kulp", "Yükselen Takoz",
    "Alçalan Takoz", "Flama", "Elmas"
]

def preprocess_image(image: Image.Image):
    img = image.convert("RGB").resize((640, 640))
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.get("/")
def read_root():
    return {"status": "YOLO ONNX Sunucusu Çalışıyor", "model_loaded": session is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Yetkisiz erişim")
    
    if session is None:
        raise HTTPException(status_code=500, detail="Model yüklenemedi")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    input_data = preprocess_image(image)
    outputs = session.run(None, {input_name: input_data})
    output = outputs[0][0]
    
    class_scores = output[4:, :]
    max_scores = np.max(class_scores, axis=0)
    max_class_ids = np.argmax(class_scores, axis=0)
    
    best_idx = np.argmax(max_scores)
    best_score = max_scores[best_idx]
    best_class_id = max_class_ids[best_idx]
    
    detections = []
    if best_score > 0.45:
        box = output[:4, best_idx].tolist()
        detections.append({
            "class_id": int(best_class_id),
            "class_name": CLASS_NAMES[best_class_id] if best_class_id < len(CLASS_NAMES) else "Bilinmeyen",
            "confidence": float(best_score),
            "box": box
        })
    
    return {"detections": detections}
