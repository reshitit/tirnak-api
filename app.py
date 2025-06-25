from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# YOLO modelini yükle (best.pt yolunu güncelle)
model = YOLO("best.pt")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    
    # YOLO tahminleri
    results = model(image)

    # Kısaca sınıf etiketlerini ve skoru JSON olarak gönder
    predictions = []
    for box in results[0].boxes.data.tolist():
        xmin, ymin, xmax, ymax, conf, class_id = box
        predictions.append({
            "box": [xmin, ymin, xmax, ymax],
            "confidence": conf,
            "class_id": int(class_id),
            "class_name": model.names[int(class_id)]
        })

    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

