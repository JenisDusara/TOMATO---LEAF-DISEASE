# apps.py
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import shutil

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

model = YOLO(r'C:\Users\Jenish\Desktop\Leaf_disease _detection\best.pt')

def upload_image():
    genre = st.radio(
        "How You Want To Upload Your Image",
        ('Browse Photos', 'Camera'))

    if genre == 'Camera':
        return st.camera_input("Take a picture")
    else:
        return st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])    
    
def preprocess_image(uploaded_image):
    pil_image = Image.open(uploaded_image)
    opencv_image = np.array(pil_image)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return opencv_image

def save_image(image, save_path):
    cv2.imwrite(save_path, image)
    st.success(f"Image saved as {save_path}")


#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "TomatoYellowCurlVirus4.JPG"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 5K rgb images of healthy and diseased crop leaves which is categorized into 8 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 500 test images is created later for prediction purpose.
                #### Content
                1. train (5000 images)
                2. test (500 images)
                3. validation (2500 images)

                """)
    st.write(model.names)



#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")

    def main():
        uploaded_image = upload_image()

        if uploaded_image is not None:
            processed_image = preprocess_image(uploaded_image)
            save_path = "processed_image.jpg"
            save_image(processed_image, save_path)

    if __name__ == "__main__":
        st.write("Image Processing and Saving Example")
        main()

    # Object Detection
    # model = YOLO(r'C:\Users\Jenish\Desktop\Object_Detection-main\best.pt')

    if st.button("Predict Disease"):
        st.snow()
        st.write("Our Prediction")
        model.predict(source='processed_image.jpg', save=True)
        st.image(r"C:\Users\Jenish\Desktop\Leaf_disease _detection\runs\detect\predict\processed_image.jpg")
        shutil.rmtree('runs')