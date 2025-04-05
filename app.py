import streamlit as st
import cv2
import numpy as np
from openai import OpenAI
import tempfile
import time

# Set page config
st.set_page_config(page_title="Food Detector & Recipe Generator", layout="wide")

# Initialize session state
if 'detected_items' not in st.session_state:
    st.session_state.detected_items = []
if 'recipe_generated' not in st.session_state:
    st.session_state.recipe_generated = False
if 'recipe_text' not in st.session_state:
    st.session_state.recipe_text = ""
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

# Load YOLO model (do this once)
@st.cache_resource
def load_yolo():
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        layer_names = net.getLayerNames()
        out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
        
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        fruit_vegetable_classes = {
            "banana", "apple", "orange", "carrot", "broccoli", 
            "cucumber", "grape", "pineapple", "tomato", "pepper"
        }
        
        return net, out_layers, classes, fruit_vegetable_classes
    except Exception as e:
        st.error(f"Failed to load YOLO model: {str(e)}")
        st.stop()

try:
    net, out_layers, classes, fruit_vegetable_classes = load_yolo()
except Exception as e:
    st.error(f"Error initializing model: {str(e)}")
    st.stop()

# Function to detect objects
def detect_objects(frame, net, out_layers, classes, fruit_vegetable_classes):
    try:
        h, w = frame.shape[:2]
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and classes[class_id] in fruit_vegetable_classes:
                    center_x, center_y, width, height = (detection[0:4] * np.array([w, h, w, h])).astype("int")
                    x, y = int(center_x - width / 2), int(center_y - height / 2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        detected_items = set()
        
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
                detected_items.add(classes[class_ids[i]])
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame, list(detected_items)
    except Exception as e:
        st.error(f"Error in object detection: {str(e)}")
        return frame, []

# Function to generate recipe
def generate_recipe(ingredients, api_key):
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = (
            f"I have the following ingredients: {', '.join(ingredients)}.\n"
            "Please create a detailed recipe that uses these ingredients. "
            "Include:\n"
            "1. Recipe name\n"
            "2. Complete list of ingredients with quantities\n"
            "3. Step-by-step cooking instructions\n"
            "4. Optional serving suggestions\n"
            "Make the recipe practical and delicious!"
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"Failed to generate recipe: {str(e)}")

def stop_webcam():
    st.session_state.webcam_active = False

st.title("🍏 Food Detector & Recipe Generator 🍳")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Get your API key from https://platform.openai.com/")
    manual_ingredients = st.text_input("Or enter ingredients manually (comma-separated)", 
                                     help="Example: apple, banana, carrot")
    
    if st.button("Generate Recipe from Manual Input") and manual_ingredients:
        if not api_key:
            st.error("Please enter your OpenAI API key")
        else:
            with st.spinner("Generating recipe..."):
                try:
                    ingredients = [x.strip() for x in manual_ingredients.split(',') if x.strip()]
                    st.session_state.recipe_text = generate_recipe(ingredients, api_key)
                    st.session_state.recipe_generated = True
                    st.session_state.detected_items = ingredients
                except Exception as e:
                    st.error(f"Error generating recipe: {str(e)}")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Food Detection")
    
    # Option to upload image or use webcam
    option = st.radio("Select input method:", ("Upload Image", "Use Webcam"), 
                     help="Detect food items from an image or your webcam")
    
    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                with st.spinner("Detecting food items..."):
                    processed_frame, detected = detect_objects(frame.copy(), net, out_layers, classes, fruit_vegetable_classes)
                    st.session_state.detected_items = detected
                
                st.image(processed_frame, caption="Processed Image", use_column_width=True)
                st.write("Detected items:", ", ".join(st.session_state.detected_items) if st.session_state.detected_items else "No food items detected")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    else:  # Webcam
        st.write("Press the button below to start webcam capture")
        
        if st.button("Start Detection"):
            st.session_state.webcam_active = True
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            stop_button = st.button("Stop Detection", on_click=stop_webcam)
            
            while cap.isOpened() and st.session_state.webcam_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                processed_frame, detected = detect_objects(frame.copy(), net, out_layers, classes, fruit_vegetable_classes)
                st.session_state.detected_items = list(set(st.session_state.detected_items + detected))
                
                frame_placeholder.image(processed_frame, caption="Live Detection", use_column_width=True)
                time.sleep(0.1)
            
            cap.release()
            if st.session_state.detected_items:
                st.write("Detection stopped. Detected items:", ", ".join(st.session_state.detected_items))
            else:
                st.write("Detection stopped. No food items detected")

with col2:
    st.header("Recipe Generator")
    
    if st.session_state.detected_items:
        st.subheader("Detected Ingredients")
        st.write(", ".join(st.session_state.detected_items))
        
        if st.button("Generate Recipe"):
            if not api_key:
                st.error("Please enter your OpenAI API key in the sidebar")
            else:
                with st.spinner("Generating recipe..."):
                    try:
                        st.session_state.recipe_text = generate_recipe(st.session_state.detected_items, api_key)
                        st.session_state.recipe_generated = True
                    except Exception as e:
                        st.error(f"Error generating recipe: {str(e)}")
    
    if st.session_state.recipe_generated:
        st.subheader("Your AI-Generated Recipe")
        st.markdown(st.session_state.recipe_text)
        
        # Add download button
        st.download_button(
            label="Download Recipe",
            data=st.session_state.recipe_text,
            file_name="generated_recipe.txt",
            mime="text/plain"
        )