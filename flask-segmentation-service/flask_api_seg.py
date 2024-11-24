from flask import Flask, request, jsonify
import torch
import numpy as np
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from PIL import Image
import io
import base64

app = Flask(__name__)

# Определение классов и размеров изображения
CLASSES = ["фон", "волосы", "кожа"]
INFER_WIDTH = 256
INFER_HEIGHT = 256

# Статистика нормализации для ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Определение устройства для вычислений
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка JIT модели
best_model = torch.jit.load('best_model.pt', map_location=DEVICE)

def get_validation_augmentation():
    """Получить аугментации для валидации."""
    test_transform = [
        albu.LongestMaxSize(max_size=INFER_HEIGHT, always_apply=True),
        albu.PadIfNeeded(min_height=INFER_HEIGHT, min_width=INFER_WIDTH, always_apply=True),
        albu.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return albu.Compose(test_transform)

def infer_image(image):
    """Получить маску на изображении с помощью модели Unet."""
    original_height, original_width, _ = image.shape

    # Применение аугментаций
    augmentation = get_validation_augmentation()
    augmented = augmentation(image=image)
    image_transformed = augmented['image']

    # Преобразование изображения в PyTorch тензор и перемещение на устройство
    x_tensor = torch.from_numpy(image_transformed).to(DEVICE).unsqueeze(0).permute(0, 3, 1, 2).float()

    # Прогон изображения через модель
    best_model.eval()
    with torch.no_grad():
        pr_mask = best_model(x_tensor)

    # Преобразование вывода в массив numpy и удаление размерности пакета
    pr_mask = pr_mask.squeeze().cpu().detach().numpy()

    # Получение класса с наивысшей вероятностью для каждого пикселя
    label_mask = np.argmax(pr_mask, axis=0)

    # Определение количества пикселей, которые будут появляться по бокам от паддингов, и их обрезка
    if original_height > original_width:
        delta_pixels = int(((original_height-original_width)/2)/original_height * INFER_HEIGHT)
        mask_cropped = label_mask[:, delta_pixels + 1 : INFER_WIDTH - delta_pixels - 1]
    elif original_height < original_width:
        delta_pixels = int(((original_width-original_height)/2)/original_width * INFER_WIDTH)
        mask_cropped = label_mask[delta_pixels + 1: INFER_HEIGHT - delta_pixels - 1, :]
    else:
        mask_cropped = label_mask

    # Изменение размера маски обратно к исходному размеру изображения
    label_mask_real_size = cv2.resize(
        mask_cropped, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    return label_mask_real_size

@app.route('/infer', methods=['POST'])
def infer():
    # Получение изображения из запроса
    file = request.files['image']
    image = np.array(Image.open(file.stream))

    # Получение маски
    mask = infer_image(image)

    # Преобразование маски в изображение для возврата
    mask_image = Image.fromarray(mask.astype(np.uint8))
    mask_bytes = io.BytesIO()
    mask_image.save(mask_bytes, format='PNG')
    mask_bytes.seek(0)

    # Кодирование маски в base64
    mask_base64 = base64.b64encode(mask_bytes.getvalue()).decode('utf-8')

    return jsonify({'mask': mask_base64})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5050)