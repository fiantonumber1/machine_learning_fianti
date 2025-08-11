import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow import keras
import keras_tuner as kt

# ======== Konfigurasi ========
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models"
DATASET_PATH = "dataset"  # dataset/Cancer dan dataset/Noncancer
TUNER_DIR = "tuner_results"
PROJECT_NAME = "cnn_cancer"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(TUNER_DIR):
    os.makedirs(TUNER_DIR)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
cnn_model_path = os.path.join(MODEL_PATH, "cnn_best_model.h5")

# ======== Fungsi Preprocessing untuk Prediksi ========
def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = img_array.astype("float32") / 255.0
    return img_array.reshape((1, 64, 64, 3))

# ======== Build Model CNN untuk Hyperparameter Tuning ========
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(64, 64, 3)))

    # Blok Conv2D
    for i in range(hp.Int("conv_blocks", 1, 2, default=1)):
        model.add(keras.layers.Conv2D(
            filters=hp.Choice(f"filters_{i}", [16, 32, 64]),
            kernel_size=hp.Choice(f"kernel_size_{i}", [3]),
            activation="relu",
            padding="same"
        ))
        model.add(keras.layers.MaxPooling2D(pool_size=2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Choice("dense_units", [32, 64]),
        activation="relu"
    ))
    model.add(keras.layers.Dropout(rate=hp.Float("dropout", 0.2, 0.4, step=0.1)))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice("learning_rate", [1e-3, 1e-4])
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ======== Endpoint Home ========
@app.route("/")
def home():
    return jsonify({"message": "Cancer Detection CNN API is Running!"})

# ======== Endpoint Train ========
@app.route("/train", methods=["POST"])
def train():
    try:
        # Data Augmentation + Load data tanpa semua ke RAM
        datagen_train = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=0.2
        )

        train_generator = datagen_train.flow_from_directory(
            DATASET_PATH,
            target_size=(64, 64),
            batch_size=16,
            class_mode="binary",
            subset="training"
        )

        val_generator = datagen_train.flow_from_directory(
            DATASET_PATH,
            target_size=(64, 64),
            batch_size=16,
            class_mode="binary",
            subset="validation"
        )

        # Hyperparameter tuning
        tuner = kt.Hyperband(
            build_model,
            objective="val_accuracy",
            max_epochs=10,
            factor=3,
            directory=TUNER_DIR,
            project_name=PROJECT_NAME,
            overwrite=False,
            executions_per_trial=1
        )

        # Checkpoint supaya tidak mulai dari nol
        checkpoint_cb = keras.callbacks.ModelCheckpoint(
            cnn_model_path,
            save_best_only=True,
            monitor="val_accuracy",
            mode="max"
        )

        tuner.search(
            train_generator,
            validation_data=val_generator,
            epochs=10,
            callbacks=[checkpoint_cb]
        )

        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save(cnn_model_path)

        return jsonify({"message": "Training CNN with hyperparameter tuning completed!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======== Endpoint Predict ========
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    try:
        if not os.path.exists(cnn_model_path):
            return jsonify({"error": "Model belum dilatih"}), 400

        model = keras.models.load_model(cnn_model_path)
        img = preprocess_image(file_path)
        pred = model.predict(img)[0][0]

        diagnosis = "Cancer" if pred >= 0.5 else "Non-Cancer"

        return jsonify({
            "Prediction_Score": float(pred),
            "Final_Diagnosis": diagnosis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
