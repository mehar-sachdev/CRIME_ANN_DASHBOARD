import streamlit as st
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from difflib import get_close_matches

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from model import CrimeANN

st.set_page_config(page_title="Crime Intelligence Dashboard", layout="wide")

st.title("🚔 Crime Intelligence & ANN Prediction System")

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Crimes_in_india_2001-2013.csv")

    df = pd.read_csv(file_path)

    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("/", "_")
    df.columns = df.columns.str.replace("&", "AND")
    df.columns = df.columns.str.replace("-", "_")

    df = df.drop(columns=["DISTRICT"], errors="ignore")
    return df

df = load_data()

# ===============================
# 🔎 CRIME SEARCH SECTION
# ===============================

st.header("🔎 Crime Search & Legal Insight")

keyword = st.text_input("Enter crime keyword (e.g., theft, murder, rape)")

punishments = {
    "MURDER": "Section 302 IPC – Punishable with death or life imprisonment.",
    "RAPE": "Section 376 IPC – Minimum 10 years imprisonment.",
    "THEFT": "Section 378 IPC – Up to 3 years imprisonment.",
    "ROBBERY": "Section 392 IPC – Up to 10 years imprisonment.",
    "RIOTS": "Section 147 IPC – Up to 2 years imprisonment.",
    "CHEATING": "Section 420 IPC – Up to 7 years imprisonment.",
    "DOWRY_DEATHS": "Section 304B IPC – Minimum 7 years imprisonment."
}

if keyword:
    keyword_upper = keyword.upper()

    # Fuzzy matching
    matches = get_close_matches(keyword_upper, df.columns, n=5, cutoff=0.4)

    if matches:
        st.success(f"Matching crimes found: {matches}")

        # Show trend
        if "YEAR" in df.columns:
            trend = df.groupby("YEAR")[matches].sum()
            st.line_chart(trend)

        # Show punishment info
        for crime in matches:
            if crime in punishments:
                st.info(f"⚖ {crime}: {punishments[crime]}")
    else:
        st.warning("No matching crime found.")

# ===============================
# 🤖 ANN PREDICTION SECTION
# ===============================

st.header("🤖 ANN Crime Prediction Model")

features = [
    'MURDER',
    'RAPE',
    'ROBBERY',
    'THEFT',
    'RIOTS',
    'CHEATING',
    'HURT_GREVIOUS_HURT',
    'DOWRY_DEATHS'
]

target = 'TOTAL_IPC_CRIMES'

df_model = df[features + [target]].copy()
X = df_model[features]
y = df_model[target]

# Sidebar controls
st.sidebar.header("⚙ Model Configuration")

hidden_layers = st.sidebar.slider("Hidden Layers", 1, 5, 2)
neurons = st.sidebar.slider("Neurons per Layer", 8, 256, 64, 8)
dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3, 0.05)

st.sidebar.header("⚙ Training Settings")

learning_rate = st.sidebar.selectbox("Learning Rate", [0.01, 0.001, 0.0001])
epochs = st.sidebar.slider("Epochs", 20, 300, 100, 10)

train_button = st.sidebar.button("🚀 Train Model")

def train_model(model, X_train, y_train, lr, epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return model, losses

if train_button:

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = CrimeANN(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        neurons=neurons,
        dropout_rate=dropout_rate
    )

    model, losses = train_model(model, X_train, y_train, learning_rate, epochs)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test)

    y_pred = scaler_y.inverse_transform(
        y_pred_scaled.detach().numpy().reshape(-1,1)
    )

    y_test_actual = scaler_y.inverse_transform(
        y_test.detach().numpy().reshape(-1,1)
    )

    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    st.subheader("📊 Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE", f"{mse:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("MAE", f"{mae:.4f}")
    col4.metric("R² Score", f"{r2:.4f}")

    st.subheader("📉 Training Loss Curve")

    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    st.pyplot(fig)

    st.success("Model Training Completed Successfully!")
