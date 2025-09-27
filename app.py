import streamlit as st
import numpy as np
import cv2
import easyocr
from ultralytics import YOLO
from PIL import Image
from main import picDetect
import pathlib

@st.cache_resource
def load_model(weight_path="best.pt"):
    return YOLO(weight_path)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

model = load_model() 
reader = load_reader()

def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
css_path = pathlib.Path("style.css")
load_css(css_path)


st.set_page_config(page_title="Traffic Sign Detector", layout="wide", page_icon='🛑')
st.title("Traffic Sign Detector 🛑")
tabs = st.tabs(["📖 About", "🧪 Try", "📊 Results"])

with tabs[0]:
    st.header("About the Project")
    st.write("""
    This project focuses on detecting and classifying traffic signs such as 
    **Speed Limits, Stop, No Parking, Pedestrian Crossings**, and more.  
    It uses **:red[YOLOv8] for object detection** and **:red[OCR] (EasyOCR)** for reading numbers inside speed limit signs.
    """)

    st.subheader("🛑 Supported Classes")

    classes = [
        "🟠 Bump"  ,
        "⛔ No Parking "  ,
        "🚫 No Stopping "  ,
        "🔄 No U-Turn "  ,
        "🚶 Pedestrian "  ,
        "🚧 Road Work "  ,
        "⚡ Speed Limit "  ,
        "🛑 Stop"
    ]
    col1, col2 = st.columns(2)

    with col1:
        st.write(classes[0])
        st.write(classes[1])
        st.write(classes[2])
        st.write(classes[3])

    with col2:
        st.write(classes[4])
        st.write(classes[5]) 
        st.write(classes[6])
        st.write(classes[7])

    st.subheader("🛠️ Technologies Used")
    st.write("""Python - Roboflow - Ultralytics YOLO - OpenCV - EasyOCR - Streamlit""")
    st.write("")
    st.subheader("👨‍💻 Team Members")
    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Abdelhalim Ahmed**  \n[LinkedIn](https://www.linkedin.com/in/abdelhalim-ahmed-720827248/)  |  [GitHub](https://github.com/Abdelhaleem1)")
    with cols[1]:
        st.markdown("**Mohannad Ashraf**  \n[LinkedIn](https://www.linkedin.com/in/mohannad-ashraf-888b24328/)  |  [GitHub](https://github.com/MohannadAshraf14)")
    with cols[2]:
        st.markdown("**Eyad Hazem**  \n[LinkedIn](https://www.linkedin.com/in/eyad-hazem-030574330)  |  [GitHub](https://github.com/Eyadhazem1)")

with tabs[1]:
    st.header("Try the Model")
    demo_tab, upload_tab = st.tabs(["🎞️ Demo Samples", "📤 Upload Your Own"])
    with demo_tab: 
        st.subheader("Demo Images")
        Demos = {
            "Stop Sign": "demo/stop.jpg",
            "Speed Limit": "demo/speed_limit.jpg", 
            "No stopping": "demo/no_stopping.jpg",
            "No Parking": "demo/no_parking.webp",
        }

        choice = st.selectbox("Choose a demo image:", list(Demos.keys()))
        dempPath = Demos[choice]

        with st.spinner("⏳ Processing image... Please wait."):
            demoImg, n, texts = picDetect(imgPath = dempPath, model=model, reader=reader)
        demoImg = cv2.cvtColor(demoImg, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        col1.image(dempPath, caption="Original", width='stretch')
        col2.image(demoImg, caption="Detection", width='stretch')
        st.subheader("Results:")
        st.write(f"Detected :green[{n}] object/s")
        for i in range(n):
            st.write(f"Detected: :green[{texts}]")
    with upload_tab:
            st.subheader("Upload Your Own Image")
            file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
            if file is not None:
                img = np.array(Image.open(file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                with st.spinner("⏳ Processing image... Please wait."):
                    output_img, n, texts = picDetect(pic=img, model=model, reader=reader)
                if (n == 0 ):
                    st.subheader(":red[Failed to detect object/s]")
                else:
                    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
                    col1, col2 = st.columns(2)
                    col1.image(file, caption="Original Image", width='stretch')
                    col2.image(output_rgb, caption="Detected Image", width='stretch')
                    st.subheader("Results:")
                    st.write(f"Detected :green[{n}] object/s")
                    for i in range(n):
                        st.write(f"Detected: :green[{texts}]")


with tabs[2]:
    st.header("Model Results & Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("mAP50", "92%") 
    col2.metric("Precision", "93%")
    col3.metric("Recall", "88%")
    st.subheader("Detection Example")
    col1, col2 = st.columns(2)
    col1.image("demo/test4.webp", caption="Original Image", width='stretch')
    col2.image(cv2.cvtColor(picDetect(imgPath="demo/test4.webp", model=model, reader=reader)[0], cv2.COLOR_BGR2RGB), caption="Detected Image", width='stretch')