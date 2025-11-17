## **UCF101 – Skeleton Action Recognition (LSTM)**

Este proyecto entrena un modelo LSTM para clasificar acciones del dataset UCF101 usando únicamente puntos clave (keypoints) 2D extraídos de las poses. El código carga un archivo .pkl con anotaciones, preprocesa los keypoints y entrena un clasificador con PyTorch.

## Cómo ejecutar

Coloca tu archivo ucf101_2d.pkl en data/. (descarga UCF101 [[2D Skeleton])](https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ucf101_2d.pkl)

Instala dependencias:

pip install torch numpy scikit-learn tqdm

Ejecuta el script:

python train.py
