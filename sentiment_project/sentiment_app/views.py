from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json  # Импортируем json для десериализации тела запроса
from django.views.decorators.csrf import csrf_exempt  # Импортируем декоратор

# Загрузка сохраненной модели и токенизатора
model_path = "../custom_model"  # Путь к вашей сохраненной модели
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

@csrf_exempt 
def predict(request):
    if request.method == "POST":
        try:
            # Получаем данные из тела запроса
            data = json.loads(request.body)  # Десериализация JSON
            text = data.get("text", "")
            if not text:
                return JsonResponse({"error": "No text provided"}, status=400)
            
            # Токенизация текста
            encodings = tokenizer(
                text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
            )
            
            # Предсказание
            with torch.no_grad():
                outputs = model(**encodings)
                logits = outputs.logits
                predicted_label = logits.argmax(axis=1).item()
                label = "negative" if predicted_label == 0 else "positive"
            
            return JsonResponse({"label": label})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    else:
        return JsonResponse({"error": "Invalid HTTP method"}, status=405)
