import streamlit as st
from PIL import Image
import cv2
import numpy as np
import requests
import io
import base64

# Определение классов и размеров изображения
CLASSES = ["фон", "волосы", "кожа"]

def adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index):
    """Корректировка значения HSV на изображении в области, где mask == index."""
    # Преобразование изображения в HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = cv2.split(image_hsv)
    
    # Применение корректировок только к области, где mask == index
    h[mask == index] = np.clip(h[mask == index] + h_adjust, 0, 179)
    s[mask == index] = np.clip(s[mask == index] + s_adjust, 0, 255)
    v[mask == index] = np.clip(v[mask == index] + v_adjust, 0, 255)
    
    # Объединение каналов HSV обратно в одно изображение
    image_hsv_adjusted = cv2.merge([h, s, v])
    
    # Преобразование изображения обратно в RGB для отображения
    image_rgb_adjusted = cv2.cvtColor(image_hsv_adjusted.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    return image_rgb_adjusted

def display_image(image):
    """Отображение изображения."""
    st.image(image, use_container_width=True)

def upload_image(label):
    """Загрузка изображения."""
    uploaded_file = st.file_uploader(label, type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image_data = np.array(Image.open(uploaded_file))
        return image_data
    return None


st.set_page_config(
    page_title="ML Обрабочик",
    page_icon='😎',
    layout="wide",
    initial_sidebar_state="expanded",)
    
def image_processing_page():

    st.title('Инструмент корректировки изображений')

    # Загрузка изображения
    image = upload_image('Загрузите изображение')

    # Проверка, что изображение загружено
    if image is not None:
        # Выбор значений для корректировки HSV
        h_adjust = st.sidebar.slider('Корректировка оттенка (H) (-179 до 179)', -179, 179, 0)
        s_adjust = st.sidebar.slider('Корректировка насыщенности (S) (-255 до 255)', -255, 255, 0)
        v_adjust = st.sidebar.slider('Корректировка освещения (V) (-255 до 255)', -255, 255, 0)

        # Выбор значения для изменения в маске с помощью выпадающего списка
        mask_value = st.sidebar.selectbox('Выберите интересующую область', CLASSES)

        # Ищем индекс значения в списке
        index = CLASSES.index(mask_value)

        # Преобразование изображения в байтовый объект
        image_pil = Image.fromarray(image)
        image_bytes = io.BytesIO()
        image_pil.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Отправка изображения на Flask API для получения маски
        response = requests.post("http://flask-segmentation-service:5050/infer", files={'image': image_bytes})

        if response.status_code == 200:
            mask_base64 = response.json()['mask']
            mask_bytes = base64.b64decode(mask_base64)
            mask = np.array(Image.open(io.BytesIO(mask_bytes)))

            # Применение корректировок HSV
            adjusted_image = adjust_hsv(image, mask, h_adjust, s_adjust, v_adjust, index)

            # Отображение исходного изображения и скорректированного изображения в двух столбцах
            col1, col2, _ = st.columns(3)
            with col1:
                display_image(image)
            with col2:
                display_image(adjusted_image)
        else:
            st.error(f"Request failed with status code {response.status_code}")

def titanic_prediction_page():

    # Заголовок приложения
    st.title("Titanic Survival Prediction")

    # Ввод данных
    st.write("Enter the passenger details:")

    # Выпадающее меню для выбора класса билета
    pclass = st.selectbox("Ticket Class (Pclass)", [1, 2, 3])

    # Текстовое поле для ввода возраста с проверкой на число
    age = st.text_input("Age", value=10)
    if not age.isdigit():
        st.error("Please enter a valid number for Age.")

    # Текстовое поле для ввода стоимости билета с проверкой на число
    fare = st.text_input("Fare", value=100)
    if not fare.isdigit():
        st.error("Please enter a valid number for Fare.")

    # Кнопка для отправки запроса
    if st.button("Predict"):
        # Проверка, что все поля заполнены
        if age.isdigit() and fare.isdigit():
            # Подготовка данных для отправки
            data = {
                "Pclass": int(pclass),
                "Age": float(age),
                "Fare": float(fare)
            }

            # Отправка запроса к Flask API
            response = requests.post("http://flask-titanic-service:5000/predict_model", data=data)

            # Проверка статуса ответа
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"Request failed with status code {response.status_code}")
        else:
            st.error("Please fill in all fields with valid numbers.")

# Основная функция для выбора страницы
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Image Processing", "Titanic Prediction"])

    if page == "Image Processing":
        image_processing_page()
    elif page == "Titanic Prediction":
        titanic_prediction_page()

if __name__ == '__main__':
    main()