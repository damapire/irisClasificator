
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Clasificador Iris (Keras + Streamlit)", layout="wide")
st.title("🌸 Clasificador Iris con Keras + Streamlit")

# -----------------------------
# Datos base
# -----------------------------
iris = load_iris()
feature_names_full = iris.feature_names  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
feature_map = [
    ("Largo sépalo (cm)", 0),
    ("Ancho sépalo (cm)", 1),
    ("Largo pétalo (cm)", 2),
    ("Ancho pétalo (cm)", 3),
]

# -----------------------------
# Sidebar: selección de variables
# -----------------------------
st.sidebar.header("🧪 Variables (features)")
default_feats = [name for name, _ in feature_map]  # por defecto: todas
selected_feats = st.sidebar.multiselect(
    "Elige las variables que quieres usar para el modelo",
    options=[name for name, _ in feature_map],
    default=default_feats,
    help="Puedes entrenar con 1 a 4 variables (sépalo y/o pétalo)."
)

if len(selected_feats) == 0:
    st.sidebar.warning("Selecciona al menos una variable para continuar.")
    st.stop()

selected_idx = [idx for name, idx in feature_map if name in selected_feats]

# -----------------------------
# Sidebar: hiperparámetros
# -----------------------------
st.sidebar.header("⚙️ Hiperparámetros")

# Arquitectura
n1 = st.sidebar.slider("Neuronas capa 1", 8, 256, 128, step=8)
n2 = st.sidebar.slider("Neuronas capa 2", 8, 256, 64, step=8)
use_third = st.sidebar.checkbox("Usar 3ra capa oculta", True)
n3 = st.sidebar.slider("Neuronas capa 3", 4, 128, 32, step=4, disabled=not use_third)

# Regularización
drop1 = st.sidebar.slider("Dropout capa 1", 0.0, 0.7, 0.3, step=0.05)
drop2 = st.sidebar.slider("Dropout capa 2", 0.0, 0.7, 0.3, step=0.05)
drop3 = st.sidebar.slider("Dropout capa 3", 0.0, 0.7, 0.0, step=0.05, disabled=not use_third)
l2_1 = st.sidebar.select_slider("L2 capa 1", options=[0.0, 1e-4, 5e-4, 1e-3, 1e-2], value=1e-3)
l2_2 = st.sidebar.select_slider("L2 capa 2", options=[0.0, 1e-4, 5e-4, 1e-3, 1e-2], value=5e-4)
l2_3 = st.sidebar.select_slider("L2 capa 3", options=[0.0, 1e-4, 5e-4, 1e-3, 1e-2], value=0.0, disabled=not use_third)

# Entrenamiento
epochs = st.sidebar.slider("Épocas", 5, 200, 50, step=5)
batch_size = st.sidebar.select_slider("Batch size", options=[8, 16, 32, 64], value=32)
val_split = st.sidebar.slider("Validation split", 0.1, 0.4, 0.2, step=0.05)
patience = st.sidebar.slider("EarlyStopping patience", 1, 20, 5)
monitor = st.sidebar.selectbox("Monitor", ["val_loss", "val_accuracy"], index=0)
test_size = st.sidebar.slider("Tamaño de test", 0.1, 0.4, 0.2, step=0.05)
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

st.sidebar.write("---")
st.sidebar.caption("Tip: puedes entrenar solo con sépalo, solo con pétalo o combinarlos.")

# -----------------------------
# Preparación de datos con features seleccionadas
# -----------------------------
X_all = iris.data[:, selected_idx]
y_all = iris.target.reshape(-1, 1)

# Escalado
scaler = StandardScaler()
X_all = scaler.fit_transform(X_all)

# One-hot para y
enc = OneHotEncoder(sparse_output=False)
y_onehot = enc.fit_transform(y_all)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_onehot, test_size=test_size, stratify=y_onehot, random_state=seed
)

# -----------------------------
# Construcción del modelo
# -----------------------------
def build_model(input_dim: int):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(n1, activation="relu",
                    kernel_regularizer=regularizers.l2(l2_1) if l2_1 else None))
    if drop1 > 0: model.add(Dropout(drop1))

    model.add(Dense(n2, activation="relu",
                    kernel_regularizer=regularizers.l2(l2_2) if l2_2 else None))
    if drop2 > 0: model.add(Dropout(drop2))

    if use_third:
        model.add(Dense(n3, activation="relu",
                        kernel_regularizer=regularizers.l2(l2_3) if l2_3 else None))
        if drop3 > 0: model.add(Dropout(drop3))

    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model(X_train.shape[1])

# -----------------------------
# Entrenamiento
# -----------------------------
early = EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)
with st.spinner("Entrenando el modelo..."):
    history = model.fit(
        X_train, y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early],
        verbose=0
    )

# -----------------------------
# Resultados de entrenamiento
# -----------------------------
st.subheader("📈 Resultados de entrenamiento")

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Pérdida (loss)**")
    fig1 = plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()
    st.pyplot(fig1)

with col2:
    st.markdown("**Exactitud (accuracy)**")
    fig2 = plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.xlabel("Época"); plt.ylabel("Accuracy"); plt.legend()
    st.pyplot(fig2)

# Evaluación en test
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)
acc = accuracy_score(y_true, y_pred)

met_left, met_right = st.columns(2)
with met_left:
    st.metric(label="Accuracy en test", value=f"{acc*100:.2f}%")
with met_right:
    st.write("**Features usadas:**", ", ".join(selected_feats))

st.markdown("**Reporte de clasificación**")
report = classification_report(y_true, y_pred, target_names=iris.target_names, zero_division=0)
st.text(report)

# Matriz de confusión
st.markdown("**Matriz de confusión**")
cm = confusion_matrix(y_true, y_pred)
fig_cm = plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix")
plt.xticks(ticks=range(len(iris.target_names)), labels=iris.target_names, rotation=45)
plt.yticks(ticks=range(len(iris.target_names)), labels=iris.target_names)
plt.xlabel("Predicción"); plt.ylabel("Real")
thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
st.pyplot(fig_cm)

st.write("---")

# -----------------------------
# Predicción interactiva (sliders)
# -----------------------------
st.subheader("🎯 Predicción interactiva")

# Rango real de cada feature para construir sliders
feature_ranges = {i: (float(iris.data[:, i].min()), float(iris.data[:, i].max())) for i in range(4)}

cols = st.columns(min(4, len(selected_idx)))
user_input = []
for k, idx in enumerate(selected_idx):
    col = cols[k % len(cols)]
    with col:
        fmin, fmax = feature_ranges[idx]
        step = (fmax - fmin) / 100.0
        label = [name for name, ii in feature_map if ii == idx][0]
        val = st.slider(label, fmin, fmax, float(np.mean(iris.data[:, idx])), step=step)
        user_input.append(val)

user_array = np.array(user_input).reshape(1, -1)
user_array_scaled = scaler.transform(user_array)
probs = model.predict(user_array_scaled, verbose=0)[0]
pred_idx = int(np.argmax(probs))
pred_name = iris.target_names[pred_idx]

st.markdown(f"**Predicción:** `{pred_name}`")
st.write("**Probabilidades:**")
prob_cols = st.columns(3)
for i, name in enumerate(iris.target_names):
    with prob_cols[i]:
        st.metric(name, f"{probs[i]*100:.1f}%")

st.caption("Mueve los deslizadores para cambiar largo/ancho de sépalo y pétalo (o las variables que hayas elegido) y ver cómo cambia la predicción.")
st.caption("Hecho con ❤️ en Streamlit + Keras.")
