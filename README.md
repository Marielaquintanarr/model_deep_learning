
# Modelo de Deep Learning para Clasificación de Acciones en Videos

## Descripción

Este proyecto implementa un modelo de deep learning basado en LSTM (Long Short-Term Memory) para clasificar acciones humanas en videos utilizando keypoints 2D extraídos del dataset UCF101.

## Requisitos
- torch
- numpy
- scikit-learn
- matplotlib
- seaborn
- pandas
- tqdm
  
## Características del Modelo

### Arquitectura

- Tipo: LSTM (Long Short-Term Memory)
- Input: Secuencias temporales de keypoints 2D
  - Shape: (MAX_FRAMES, V*2) donde V es el número de keypoints
  - MAX_FRAMES: 300 (configurable)
- Capas:
  - LSTM: 1 capa con 128 unidades ocultas
  - Linear: Capa de clasificación final
- Output: Probabilidades sobre 5 clases seleccionadas

### Parámetros del Modelo

- Input dimension: V*2 (depende del número de keypoints)
- Hidden units: 128
- Número de clases: 5 

### Preprocesamiento

1. Selección de persona: Se toma solo la primera persona 
2. Reshape: Conversión a formato (T, V*2)
3. Normalización: 
   - Centrado: resta de la media
   - Estandarización: división por desviación estándar

### Hiperparámetros de Entrenamiento

- Batch size: 16
- Epochs: 5
- Learning rate: 1e-3
- Loss function: CrossEntropyLoss
- Train/Val split: 80/20

## Estructura del Proyecto
- Modelo_de_deep_learning.ipynb # Notebook principal
- [ucf101_2d.pkl](https://drive.google.com/file/d/1pnJVwb1c7q8mQxIkPhZuUTV3xBmc1VLO/view?usp=sharing) # Dataset con keypoints 


## Uso

### Inferencia

Correr celda 19 del notebook.

## Métricas y Evaluación

El modelo genera las siguientes métricas:

### Métricas por Época
- Train Loss: Pérdida en conjunto de entrenamiento
- Val Loss: Pérdida en conjunto de validación
- Val Accuracy: Precisión global
- Val Precision: Precisión promedio ponderada
- Val Recall: Recall promedio ponderado
- Val F1-Score: F1-score promedio ponderado

### Gráficas
1. Gráfica de pérdida: Evolución de train/val loss por época
2. Gráfica de métricas: Evolución de accuracy, precision, recall, F1
3. Matriz de confusión: Visualización de predicciones vs. reales


## Resultados del Modelo Base
Los resultados se guardan en:
- `metricas_entrenamiento.json`

  
