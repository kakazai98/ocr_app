import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import json

IMG_WIDTH, IMG_HEIGHT = 32, 32  # use same size you trained with


@st.cache_resource
def load_ocr_model():
    model = load_model("ocr_model.keras")

    # Load class mapping: label -> index, then invert it to index -> label
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)  # e.g. {"A": 0, "B": 1, "C": 2, ...}

    idx_to_label = {v: k for k, v in class_indices.items()}
    return model, idx_to_label


def preprocess_image(img: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into the format expected by the model:
    - RGB
    - resized to 32x32
    - normalized to [0, 1]
    - shape: (1, 32, 32, 3)
    """
    img = img.convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    st.set_page_config(page_title="OCR Character Classifier")
    st.title("OCR Character Classifier")
    st.write(
        "Upload an image containing a **single character** "
        "(digit / letter / punctuation) and I will predict it."
    )

    model, idx_to_label = load_ocr_model()

    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=200)

        if st.button("Predict"):
            input_array = preprocess_image(image)
            preds = model.predict(input_array)
            predicted_index = int(np.argmax(preds[0]))
            predicted_label = idx_to_label.get(predicted_index, "Unknown")

            st.markdown(f"### ✅ Predicted character: **{predicted_label}**")

            # Optional: show top probabilities
            with st.expander("Show prediction probabilities"):
                probs = preds[0]
                labels_probs = sorted(
                    [(idx_to_label[i], float(probs[i])) for i in range(len(probs))],
                    key=lambda x: x[1],
                    reverse=True
                )
                st.write(labels_probs[:10])  # top 10 classes


if __name__ == "__main__":
    main()
