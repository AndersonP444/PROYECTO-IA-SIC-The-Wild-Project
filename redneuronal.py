import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# ======================================
# CARGAR Y AUMENTAR DATOS
# ======================================
dataset_url = "https://github.com/AndersonP444/PROYECTO-IA-SIC-The-Wild-Project/raw/refs/heads/main/password_dataset_final.csv"
df = pd.read_csv(dataset_url)

# Añadir más contraseñas débiles y patrones comunes
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
    {"password": "qwerty123", "strength": 0},
    {"password": "Admin1234", "strength": 0},
    {"password": "Welcome1", "strength": 0},
]

weak_df = pd.DataFrame(weak_passwords)
df = pd.concat([df, weak_df], ignore_index=True)

# ======================================
# PREPROCESAMIENTO MEJORADO
# ======================================
COMMON_NAMES = set(["pepito", "juan", "maria", "pedro", "ana", "luis", 
                   "carlos", "laura", "sofia", "diego", "password", "qwerty",
                   "123456", "admin", "iloveyou", "welcome"])

def has_consecutive(password):
    for i in range(len(password)-1):
        if abs(ord(password[i]) - ord(password[i+1])) == 1:
            return 1
    return 0

def has_repeated(password):
    for i in range(len(password)-2):
        if password[i] == password[i+1] == password[i+2]:
            return 1
    return 0

def preprocesar_dataset(df):
    X = []
    for _, row in df.iterrows():
        pwd = row["password"].lower()
        features = [
            len(row["password"]),                        # 1. Longitud total
            sum(1 for c in pwd if c.isupper()),          # 2. Conteo de mayúsculas
            sum(1 for c in pwd if c.isdigit()),          # 3. Cantidad de dígitos
            sum(1 for c in pwd if c in "!@#$%^&*()"),    # 4. Cantidad de símbolos
            int(any(name in pwd for name in COMMON_NAMES)),  # 5. Palabras débiles
            int("123" in pwd or "456" in pwd or "789" in pwd),  # 6. Secuencias numéricas
            has_consecutive(pwd),                        # 7. Caracteres consecutivos
            has_repeated(pwd),                           # 8. Caracteres repetidos
            sum(ord(c) for c in pwd)/len(pwd) if len(pwd) > 0 else 0,  # 9. Entropía simple
            int(pwd == pwd[::-1]),                       # 10. Es palíndromo
        ]
        X.append(features)
    
    X = np.array(X)
    
    # Normalización avanzada
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    y = df["strength"].values
    return X, y, scaler

X, y, scaler = preprocesar_dataset(df)

# ======================================
# BALANCEO DE DATOS
# ======================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Aplicar SMOTE para balancear clases
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ======================================
# ARQUITECTURA DE LA RED NEURONAL
# ======================================
def crear_modelo():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(10,), kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
        BatchNormalization(),
        
        Dense(3, activation='softmax')
    ])
    
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = crear_modelo()

# ======================================
# ENTRENAMIENTO AVANZADO
# ======================================
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# ======================================
# EVALUACIÓN FINAL
# ======================================
model.load_weights('best_model.h5')  # Cargar mejor versión del modelo

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f'\nPrecisión final en validación: {val_acc * 100:.2f}%')

# Guardar modelo y escalador
model.save("password_strength_model_pro.h5")
import joblib
joblib.dump(scaler, 'scaler_pro.save')
print("Modelo y escalador guardados!")
