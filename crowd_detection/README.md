# Crowd Detection
## Описание проекта

Этот проект обеспечивает обнаружение и анализ толпы с использованием:
- Faster R-CNN ResNet50
- Статистический анализ качества обнаружения

Ключевые фичи:
- Обработка видео с покадровым анализом
- Визуализация распределения скоров уверенности
- Статистика размеров bbox
- Кроссплатформенная совместимость

## Установка

### 1. Клонируем репозиторий
```bash
git clone https://github.com/KarenYer/pet_projects/tree/main/crowd_detection
cd crowd_detection
```
Рекомендуем использовать виртуальное окружение, дабы установка пакетов была только в рамках проекта.
### 2. Создание и запуск виртуального окружения (опционально)
```bash
python -m venv venv

# Активируем
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```
### 3. Устанавливаем все зависимости
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### 4. Запуск проекта
```bash
python main.py
```
