'''
Давайте создадим простое Flask API с тремя ручками: одна для предсказания выживания (/predict), 
другая для получения количества сделанных запросов (/stats), и третья для проверки работы API (/health).

Шаг 1: Установка необходимых библиотек
Убедитесь, что у вас установлены необходимые библиотеки:
pip install flask scikit-learn pandas

Шаг 2: Создание flask_api.py
Шаг 3: Запустите ваше Flask приложение: python flask_api.py
Шаг 4: Тестирование API
Теперь вы можете протестировать ваше API с помощью curl или любого другого инструмента для отправки HTTP-запросов.

Проверка работы API (/health)
curl -X GET http://127.0.0.1:5000/health
curl -X GET http://127.0.0.1:5000/stats
curl -X POST http://127.0.0.1:5000/predict_model -d "Pclass=3&Age=22.0&Fare=7.2500"
'''


from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Загрузка модели из файла pickle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Счетчик запросов
request_count = 0

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({"request_count": request_count})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"})


@app.route('/predict_model', methods=['POST'])
def predict_model():
    global request_count
    request_count += 1

    # Получение данных из запроса
    data = request.form

    # Проверка наличия необходимых ключей в данных
    required_keys = ['Pclass', 'Age', 'Fare']
    if not all(key in data for key in required_keys):
        return jsonify({"error": "Missing required keys"}), 400

    # Создание DataFrame из данных
    new_data = pd.DataFrame({
        'Pclass': [int(data['Pclass'])],
        'Age': [float(data['Age'])],
        'Fare': [float(data['Fare'])]
    })

    # Предсказание
    predictions = model.predict(new_data)

    # Преобразование результата в человеко-читаемый формат
    result = "Survived" if predictions[0] == 1 else "Not Survived"

    return jsonify({"prediction": result})



if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)