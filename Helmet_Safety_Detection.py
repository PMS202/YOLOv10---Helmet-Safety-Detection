import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLOv10
MODEL = r"D:\Hoc\AIO2024\YOLOv10---Helmet-Safety-Detection\yolov10\runs\detect\train\weights\best.pt"


def process_image(image):
    image = cv2.resize()
    model = YOLOv10(MODEL)
    CONF_THRESHOLD = 0.3
    results = model.predict(source=IMAGE_URL,
                            imgsz=IMG_SIZE,
                            conf=CONF_THRESHOLD)
    return results


def main():
    """Main function for the image object detection app."""

    st.title("Object Detection for Images")

    uploaded_file = st.file_uploader(
        "Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image")

        image = Image.open(uploaded_file)
        image = np.array(image)

        # Assuming process_image and annotate_image functions exist
        detections = process_image(image)
        processed_image = annotate_image(image, detections)

        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
