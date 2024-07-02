import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLOv10
MODEL = r"D:\Hoc\AIO2024\YOLOv10---Helmet-Safety-Detection\yolov10\runs\detect\train\weights\best.pt"


def predict(image):
    model = YOLOv10(MODEL)
    CONF_THRESHOLD = 0.3
    IMAGE_URL = image
    IMG_SIZE = 640
    results = model.predict(source=IMAGE_URL,
                            imgsz=IMG_SIZE,
                            conf=CONF_THRESHOLD)
    return results


def streamlit():
    """Main function for the image object detection app."""

    st.title("Object Detection for Images")

    uploaded_file = st.file_uploader(
        "Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")
        image = Image.open(uploaded_file,)
        # Assuming process_image and annotate_image functions exist
        detections = predict(image)
        annotated_img = detections[0].plot()
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        st.image(annotated_img, caption="Processed Image")


if __name__ == "__main__":
    streamlit()
