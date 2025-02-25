import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Cargar el dataset desde GitHub
dataset_url = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/main/password_dataset_500.csv"
df = pd.read_csv(dataset_url)

# Añadir contraseñas débiles con estructuras comunes
weak_passwords = [
    {"password": "Pepito123", "strength": 0},
    {"password": "Juan456", "strength": 0},
    {"password": "Maria789", "strength": 0},
    {"password": "Pedro123", "strength": 0},
    {"password": "Ana456", "strength": 0},
    {"password": "Luis789", "strength": 0},
    {"password": "Carlos123", "strength": 0},
    {"password": "Laura456", "strength": 0},
    {"password": "Sofia789", "strength": 0},
    {"password": "Diego123", "strength": 0},
]

# Convertir a DataFrame y concatenar con el dataset original
weak_df = pd.DataFrame(weak_passwords)
df = pd.concat([df, weak_df], ignore_index=True)

# Preprocesar el dataset
def preprocesar_dataset(df):
    # Extraer características
    X = np.array([[
        len(row["password"]),  # Longitud de la contraseña
        int(any(c.isupper() for c in row["password"])),  # Contiene mayúsculas
        int(any(c.isdigit() for c in row["password"])),  # Contiene números
        int(any(c in "!@#$%^&*()" for c in row["password"])),  # Contiene símbolos
        int(row["password"].lower() in ["pepito", "juan", "maria", "pedro", "ana", "luis", "carlos", "laura", "sofia", "diego"]),  # Nombres comunes
        int("123" in row["password"] or "456" in row["password"] or "789" in row["password"])  # Secuencias simples
    ] for _, row in df.iterrows()])
    
    # Extraer etiquetas
    y = df["strength"].values
    
    # Codificar las etiquetas (si es necesario)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y, label_encoder

# Preprocesar el dataset
X, y, label_encoder = preprocesar_dataset(df)

# Dividir el dataset en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de red neuronal mejorado
def crear_modelo():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(6,), kernel_regularizer=l2(0.01)),  # Capa oculta con 128 neuronas y regularización L2
        Dropout(0.3),  # Dropout para evitar sobreajuste
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta con 64 neuronas
        Dropout(0.3),  # Dropout
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta con 32 neuronas
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),  # Capa oculta con 16 neuronas
        Dense(3, activation='softmax')  # Capa de salida con 3 neuronas (para 3 clases: débil, media, fuerte)
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear el modelo
model = crear_modelo()

# Early Stopping para evitar sobreajuste
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar el modelo
history = model.fit(
    X_train, y_train,
    epochs=100,  # Aumentamos el número de épocas
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=1
)

# Guardar el modelo entrenado
model.save("password_strength_model.h5")
print("Modelo entrenado y guardado como 'password_strength_model.h5'.")

# Evaluar el modelo en el conjunto de validación
loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Precisión en el conjunto de validación: {accuracy * 100:.2f}%")
