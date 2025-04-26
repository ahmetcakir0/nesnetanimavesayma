import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict

# Sayfa ayarları
st.set_page_config(page_title="Nesne Tanıma", layout="wide")

# Başlık
st.title("🧠 Otomatik Nesne Tanıma Uygulaması")

# Sidebar ayarları
with st.sidebar:
    st.header("⚙️ Ayarlar")
    confidence_threshold = st.slider("Güven Eşiği", 0.0, 1.0, 0.5, 0.01)

# Görsel yükleme
uploaded_file = st.file_uploader("Görsel seçin", type=["jpg", "jpeg", "png"])

# Model yükle
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  # Daha güçlü bir model kullanıldı

model = load_model()

# Ana işlem
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Nesne tanıma
    results = model(img_array)[0]
    annotated = img_array.copy()

    object_counts = defaultdict(int)
    detected_names = []

    for box in results.boxes:
        conf = float(box.conf)
        cls = int(box.cls)
        if conf < confidence_threshold:
            continue

        label = model.names[cls].capitalize()  # Etiketin ilk harfini büyük yap
        object_counts[label] += 1
        detected_names.append(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    with col2:
        st.image(annotated, caption="Tespit Sonucu", use_column_width=True)

        st.subheader("📋 Tespit Edilen Nesneler")
        if object_counts:
            for label, count in object_counts.items():
                st.write(f"- **{label}**: {count} adet")
        else:
            st.info("Hiçbir nesne tespit edilemedi.")
