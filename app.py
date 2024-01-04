import streamlit as st
from PIL import Image
import io
import os
from detection import main
import base64

st.title("Video Analytics")

# File uploader for video input
video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if video_file is not None:

    # save the uploaded video
    with open(video_file.name, "wb") as videos:
        videos.write(video_file.read())

    # Perform object detection on the uploaded video
    st.info("Detecting objects. Please wait...")
    video_bytes = video_file.read()
    np_video = np.frombuffer(video_bytes, dtype=np.uint8)
    video_cap = cv2.VideoCapture()
    video_cap.open(video_file.name)
    detection_data = main(video_file.name, "Output.avi")

    # Create a DataFrame with detection information
    # detection_df = pd.DataFrame(detection_data, columns=["Class", "Confidence", "Box"])
    st.write("Object Detection Results:")
    st.write(detection_data)


    # Save the DataFrame to Excel
    filename = "Output.csv"
    detection_data.to_csv(filename, index=False)

    st.markdown(f"Download [CSV results](data:file/csv;base64,{base64.b64encode(open(filename, 'rb').read()).decode()})")

    video_file_name = "Output.avi"
    result_video_bytes = open(video_file_name, "rb").read()
    
    st.markdown(f"**Download Video:** [Output Video](data:video/avi;base64,{base64.b64encode(result_video_bytes).decode()})")

    video_cap.release()

    if os.path.exists(video_file.name):
        os.remove(video_file.name)

    st.cache_data.clear()
