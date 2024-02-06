import streamlit as st
import numpy as np
import cv2
from PIL import Image

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_brightness(img, rect, max_brightness=False):
    x, y, w, h = rect
    cropped_img = img[y:y+h, x:x+w]
    if max_brightness:
        return np.max(cropped_img)
    else:
        return np.mean(cropped_img)

def draw_rectangles(img, rects, colors, thickness=10, texts=None, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=6, font_color=(255, 255, 255), font_thickness=10):
    img_copy = img.copy()
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), colors[i], thickness)
        if texts and texts[i]:
            (text_width, text_height), _ = cv2.getTextSize(texts[i], font, font_scale, font_thickness)
            cv2.putText(img_copy, texts[i], (x + w + 15, y + text_height), font, font_scale, colors[i], font_thickness)
    return img_copy

st.title("ELLA : KIM-1 and NGAL quantification")

uploaded_file = st.file_uploader("upload or capture a image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img)
    st.image(img_array, caption="uploaded", use_column_width=True)
    
    gray_img = convert_to_grayscale(img_array)
    st.image(gray_img, caption="gray", use_column_width=True)

    st.subheader("ROI Settings")
    rect_count = 5
    rects = []
    w = st.number_input(f"ROI width:", key=f"w", min_value=0, value=350, step=1)
    w = st.slider(f"ROI width slider:", key=f"w_slider", min_value=0, max_value=1000, value=w)
    h = st.number_input(f"ROI height:", key=f"h", min_value=0, value=200, step=1)
    h = st.slider(f"ROI height slider:", key=f"h_slider", min_value=0, max_value=1000, value=h)
    x = st.number_input(f"ROI column point:", key=f"x", min_value=0, value=900, step=1)
    x = st.slider(f"ROI column point slider:", key=f"x_slider", min_value=0, max_value=img_array.shape[1]-w, value=x)
    y = st.number_input(f"ROI row point:", key=f"y", min_value=0, value=500, step=1)
    y = st.slider(f"ROI row point slider:", key=f"y_slider", min_value=0, max_value=img_array.shape[0]-h, value=y)
    space = st.number_input(f"ROI space", key=f"s", min_value=0, value=280, step=1)
    space = st.slider(f"ROI space slider:", key=f"s_slider", min_value=0, max_value=1000, value=space)
    for i in range(rect_count):
        rects.append((x, y+space*i, w, h))

    colors = [(255, 255, 255)] * rect_count
    colors[1] = (255, 255, 255)
    colors[3] = (255, 255, 255)

    texts = [None] * rect_count
    texts[1] = "Control"
    texts[3] = "Test"

    img_with_rects = draw_rectangles(gray_img, rects, colors, texts=texts)
    st.image(img_with_rects, caption="ROI ROIs", use_column_width=True)
    
    st.subheader("Option Selection")
    options = ['KIM-1','NGAL']
    selected_option = st.select_slider("Select KIM-1 or  NGAL", options=options)
    st.write("The selected option :mag: is",selected_option)

    st.subheader("Results")
    if st.button("Calculate Brightness using Gray image"):
        if selected_option == "KIM-1":
            Bottom = 0.2624
            Hillslope = 1.42
            Top = 0.9089
            Ec50 = 4.834
            Logec50 = 0.6843
            Span = 0.6465
        elif selected_option == "NGAL":
            Bottom = 0.2958
            Hillslope = 0.71
            Top = 1.092
            Ec50 = 3.71
            Logec50 = 0.5694
            Span = 0.7966

        brightness_mean = [get_brightness(gray_img, rect) for rect in rects]
        brightness_max = [get_brightness(gray_img, rect, max_brightness=True) for rect in rects]
        background_brightness = np.mean([brightness_max[0], brightness_max[2], brightness_max[4]])
        max_brightness_2 = get_brightness(gray_img, rects[1], max_brightness=True)
        max_brightness_4 = get_brightness(gray_img, rects[3], max_brightness=True)
        true_CL = max_brightness_2-background_brightness
        true_TL = max_brightness_4-background_brightness
        st.write(f"Value of Background (mean of max): {background_brightness:.2f}")
        st.write(f"Measured MAX CL: {max_brightness_2}")
        st.write(f"Measured MAX TL: {max_brightness_4}")
        st.write(f"True CL: {true_CL}")
        st.write(f"True TL: {true_TL}")     
        # 
        Y = true_TL / (true_TL + true_CL)
        st.write(f"R [=TL/(TL+CL)]: {Y:.2f}")
        # 농도 구하기
        result_concentration= (-Ec50**Hillslope*(Y-Bottom)/(Y-Top))**(1/Hillslope)    
        st.write(f"Concentraion of {selected_option}: {result_concentration} ng/ml")
        
        for i, b in enumerate(brightness_mean):
            st.write(f"measured brightness of {i+1} (mean): {b:.2f}")
        for i, b in enumerate(brightness_max):
            st.write(f"measured brightness of {i+1} (max): {b:.2f}")