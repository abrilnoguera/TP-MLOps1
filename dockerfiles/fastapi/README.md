# 🏠 API de Predicción de Ocupación Airbnb - Buenos Aires

API REST desarrollada con FastAPI para predecir la ocupación de propiedades Airbnb en Buenos Aires utilizando Machine Learning.

## 📋 Descripción

Esta API utiliza un modelo de Random Forest entrenado con datos históricos de Airbnb para predecir si una propiedad tendrá alta o baja ocupación basándose en sus características.

## 🚀 Instalación y Ejecución

### Opción 1: Usando Docker (Recomendado)

```bash
# Construir la imagen
docker build -t fastapi-airbnb .

# Ejecutar el contenedor
docker run -p 8000:8000 fastapi-airbnb
```

### Opción 2: Ejecución Local

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar la aplicación
uvicorn app:app --host 0.0.0.0 --port 8000
```

## 📚 Endpoints Disponibles

### 1. Endpoint Principal
- **URL**: `GET /`
- **Descripción**: Mensaje de bienvenida y estado de la API
- **Respuesta**:
```json
{
    "message": "Welcome to the Airbnb Occupancy Prediction API for Buenos Aires"
}
```

### 2. Predicción de Ocupación
- **URL**: `POST /predict`
- **Descripción**: Predice la ocupación de una propiedad basándose en sus características
- **Content-Type**: `application/json`

#### Ejemplo de Solicitud:
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "latitude": -34.6037,
    "longitude": -58.3816,
    "property_type": "Entire rental unit",
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bedrooms": 2.0,
    "beds": 2.0,
    "price": 150.0,
    "minimum_nights": 2,
    "maximum_nights": 30,
    "number_of_reviews": 45,
    "review_scores_rating": 4.8,
    "review_scores_accuracy": 4.9,
    "review_scores_cleanliness": 4.7,
    "review_scores_checkin": 4.8,
    "review_scores_communication": 4.9,
    "review_scores_location": 4.6,
    "review_scores_value": 4.5,
    "host_listings_count": 3.0,
    "host_total_listings_count": 3.0,
    "instant_bookable": 1,
    "host_is_superhost": 0,
    "city": "Buenos Aires"
}'
```

#### Respuesta Exitosa:
```json
{
    "prediction": 1,
    "occupancy_level": "High Occupancy"
}
```

## ✅ Validaciones de Entrada

La API implementa validaciones estrictas basadas en los datos de entrenamiento:

### 📍 Coordenadas Geográficas
- **Latitud**: Entre -34.68 y -34.53 (área de Buenos Aires)
- **Longitud**: Entre -58.53 y -58.36 (área de Buenos Aires)

### 🏘️ Tipo de Propiedad (property_type)
Valores válidos incluyen:
- "Entire rental unit"
- "Entire home/apt"
- "Private room"
- "Shared room"
- Y otros 59 tipos específicos validados por enum

### 🛏️ Tipo de Habitación (room_type)
- "Entire home/apt"
- "Private room"
- "Shared room"
- "Hotel room"

### 👥 Capacidad y Espacio
- **accommodates**: 1-10 huéspedes
- **bedrooms**: 0.0-10.0 habitaciones
- **beds**: 0.0-10.0 camas

### 💰 Precios y Políticas
- **price**: $0.23 - $90,927.88 USD por noche
- **minimum_nights**: 1-730 noches mínimas
- **maximum_nights**: 1-99,999 noches máximas

### ⭐ Reseñas y Calificaciones
- **number_of_reviews**: 0-992 reseñas
- **review_scores_***: Calificaciones entre 0.0-5.0 (algunas entre 1.0-5.0)

### 🏠 Información del Anfitrión
- **host_listings_count**: 1.0-670.0 propiedades
- **host_total_listings_count**: 1.0-2,542.0 propiedades totales

### 🔧 Características Binarias
- **instant_bookable**: 0 (No) o 1 (Sí)
- **host_is_superhost**: 0 (No) o 1 (Sí)

### 🌎 Ubicación
- **city**: Solo "Buenos Aires" es válido

## 🚫 Manejo de Errores

### Error de Validación (422)
```json
{
    "detail": [
        {
            "type": "enum",
            "loc": ["body", "property_type"],
            "msg": "Input should be 'Entire rental unit', 'Entire home/apt', ...",
            "input": "Invalid Property Type"
        }
    ]
}
```

### Error de Rango (422)
```json
{
    "detail": [
        {
            "type": "greater_equal",
            "loc": ["body", "latitude"],
            "msg": "Input should be greater than or equal to -34.68",
            "input": -35.0
        }
    ]
}
```

## 📖 Documentación Interactiva

Una vez que la API esté ejecutándose, puedes acceder a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Ejemplos de Prueba

### Predicción Válida
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "latitude": -34.6037,
    "longitude": -58.3816,
    "property_type": "Entire rental unit",
    "room_type": "Entire home/apt",
    "accommodates": 4,
    "bedrooms": 2.0,
    "beds": 2.0,
    "price": 150.0,
    "minimum_nights": 2,
    "maximum_nights": 30,
    "number_of_reviews": 45,
    "review_scores_rating": 4.8,
    "review_scores_accuracy": 4.9,
    "review_scores_cleanliness": 4.7,
    "review_scores_checkin": 4.8,
    "review_scores_communication": 4.9,
    "review_scores_location": 4.6,
    "review_scores_value": 4.5,
    "host_listings_count": 3.0,
    "host_total_listings_count": 3.0,
    "instant_bookable": 1,
    "host_is_superhost": 0,
    "city": "Buenos Aires"
}'
```

### Prueba de Validación (Datos Inválidos)
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
    "latitude": -35.0,
    "longitude": -58.3816,
    "property_type": "Invalid Type",
    "room_type": "Entire home/apt",
    "accommodates": 15,
    "city": "Madrid"
}'
```

## 🔧 Tecnologías Utilizadas

- **FastAPI**: Framework web moderno y rápido
- **Pydantic**: Validación de datos y serialización
- **scikit-learn**: Modelo de Machine Learning
- **pandas**: Manipulación de datos
- **uvicorn**: Servidor ASGI
- **Docker**: Containerización

## 📊 Modelo de ML

El modelo utiliza un pipeline de scikit-learn que incluye:
- Preprocesamiento de datos
- Random Forest Classifier
- Predicción binaria: 0 (Baja Ocupación) / 1 (Alta Ocupación)

## 🎯 Interpretación de Resultados

- **prediction: 0** + **occupancy_level: "Low Occupancy"** = Baja probabilidad de ocupación
- **prediction: 1** + **occupancy_level: "High Occupancy"** = Alta probabilidad de ocupación
