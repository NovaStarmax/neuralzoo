import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import get_metrics

st.set_page_config(page_title="NeuralZOO — Métriques", layout="wide")
st.title("Performance du modèle CNN")

with st.spinner("Chargement des métriques..."):
    try:
        data = get_metrics()
    except Exception as e:
        st.error(f"Impossible de charger les métriques : {e}")
        st.stop()

st.metric("Accuracy globale (test set)", f"{data['accuracy']:.2%}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Matrice de confusion")
    labels = data["class_labels"]
    cm = data["confusion_matrix"]
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale="Blues",
        text_auto=True,
        labels={"x": "Prédit", "y": "Réel"},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Rapport de classification")
    report = data["classification_report"]
    rows = []
    for cls in labels:
        m = report[cls]
        rows.append(
            {
                "Classe": cls,
                "Précision": f"{m['precision']:.3f}",
                "Rappel": f"{m['recall']:.3f}",
                "F1": f"{m['f1-score']:.3f}",
                "Support": int(m["support"]),
            }
        )
    st.dataframe(pd.DataFrame(rows).set_index("Classe"), use_container_width=True)

st.divider()
st.subheader("Courbes d'entraînement")

history = data["history"]
epochs = list(range(1, len(history["train_loss"]) + 1))

tab_loss, tab_acc = st.tabs(["Loss", "Accuracy"])

with tab_loss:
    df_loss = pd.DataFrame(
        {"Epoch": epochs, "Train": history["train_loss"], "Validation": history["val_loss"]}
    ).set_index("Epoch")
    st.line_chart(df_loss)

with tab_acc:
    if "train_acc" in history and "val_acc" in history:
        df_acc = pd.DataFrame(
            {"Epoch": epochs, "Train": history["train_acc"], "Validation": history["val_acc"]}
        ).set_index("Epoch")
        st.line_chart(df_acc)
    else:
        st.info("Données d'accuracy non disponibles.")
