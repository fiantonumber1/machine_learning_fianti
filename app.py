import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, jsonify
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from werkzeug.utils import secure_filename

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi upload folder
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models"
DATASET_PATH = "dataset"  # Folder dataset lokal
POS_PATH = os.path.join(DATASET_PATH, "Cancer")
NEG_PATH = os.path.join(DATASET_PATH, "Noncancer")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model dan scaler jika tersedia
scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
svm_model_path = os.path.join(MODEL_PATH, "svm_model.pkl")
nb_model_path = os.path.join(MODEL_PATH, "nb_model.pkl")

if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
if os.path.exists(svm_model_path):
    svm_model = joblib.load(svm_model_path)
if os.path.exists(nb_model_path):
    nb_model = joblib.load(nb_model_path)

def preprocess_image(image_path):
    """Memproses gambar sebelum prediksi."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid atau tidak bisa dibaca.")
    
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    features = scaler.transform([features])
    
    return features

def load_images_from_folder(folder, label):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

@app.route("/")
def home():
    return jsonify({"message": "Cancer Detection API is Running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """API untuk memprediksi gambar kanker atau tidak."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        features = preprocess_image(file_path)
        svm_prediction = svm_model.predict(features)[0]
        nb_prediction = nb_model.predict(features)[0]

        result = {
            "SVM_Prediction": "Cancer" if svm_prediction == 1 else "Non-Cancer",
            "Naive_Bayes_Prediction": "Cancer" if nb_prediction == 1 else "Non-Cancer",
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
@app.route("/train", methods=["POST"])
def train():
    """Endpoint untuk melatih model"""
    try:
        # Load dataset
        pos_path = r"datasets\cancer"
        neg_path = r"datasets\noncancer"

        def load_images_from_folder(folder, label):
            images = []
            labels = []
            for filename in os.listdir(folder):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.resize(img, (128, 128))
                    images.append(img)
                    labels.append(label)
            return images, labels

        positive_images, positive_labels = load_images_from_folder(pos_path, label=1)
        negative_images, negative_labels = load_images_from_folder(neg_path, label=0)

        images = np.array(positive_images + negative_images)
        labels = np.array(positive_labels + negative_labels)

        # Ekstraksi fitur HOG
        hog_features = [hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys') for img in images]
        hog_features = np.array(hog_features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Simpan scaler
        joblib.dump(scaler, os.path.join(MODEL_PATH, "scaler.pkl"))

        # Train SVM
        svm_model = SVC(kernel="linear", random_state=42)
        svm_model.fit(X_train, y_train)
        joblib.dump(svm_model, os.path.join(MODEL_PATH, "svm_model.pkl"))

        # Train Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        joblib.dump(nb_model, os.path.join(MODEL_PATH, "nb_model.pkl"))

        return jsonify({"message": "Training completed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
