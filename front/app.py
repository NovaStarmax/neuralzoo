import base64

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

from config import ANIMAL_CLASSES, DEFAULT_N_IMAGES, MAX_N_IMAGES
from utils import get_sample, predict

st.set_page_config(page_title="NeuralZOO", layout="wide")
st.title("NeuralZOO — Classification d'animaux")

with st.sidebar:
    st.header("Filtres")
    selected_classes = st.multiselect(
        "Classes animales",
        ANIMAL_CLASSES,
        placeholder="Toutes les classes",
    )
    n_images = st.slider("Nombre d'images", min_value=1, max_value=MAX_N_IMAGES, value=DEFAULT_N_IMAGES)
    load_btn = st.button("Charger les images", type="primary", use_container_width=True)

if load_btn:
    with st.spinner("Chargement des images..."):
        try:
            images = get_sample(selected_classes, n_images)
            st.session_state["images"] = images
            st.session_state["prediction"] = None
        except Exception as e:
            st.error(f"Erreur lors du chargement : {e}")

images = st.session_state.get("images", [])

if not images:
    st.info("Utilisez le panneau latéral pour charger des images.")
    st.stop()

st.subheader("Grille d'images — cliquez sur une image pour la classifier")

cols_per_row = 3
rows = [images[i : i + cols_per_row] for i in range(0, len(images), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for col, img_data in zip(cols, row):
        with col:
            img_bytes = base64.b64decode(img_data["base64"])
            st.image(img_bytes, caption=f"Vrai : {img_data['true_label']}", use_container_width=True)
            if st.button("Classifier", key=f"btn_{img_data['id']}"):
                with st.spinner("Inférence CNN..."):
                    try:
                        result = predict(img_data["id"])
                        st.session_state["prediction"] = result
                        st.session_state["prediction_image"] = img_data
                    except Exception as e:
                        st.error(f"Erreur : {e}")

prediction = st.session_state.get("prediction")
if prediction:
    pred_image = st.session_state.get("prediction_image", {})
    st.divider()
    st.subheader("Résultat de la prédiction")

    left, right = st.columns([1, 2])
    with left:
        if pred_image.get("base64"):
            st.image(base64.b64decode(pred_image["base64"]), use_container_width=True)
        correct = prediction["is_correct"]
        result_text = "Correct" if correct else "Incorrect"
        color = "green" if correct else "red"
        st.markdown(
            f"**Prédit :** {prediction['predicted_class']}  \n"
            f"**Réel :** {prediction['true_label']}  \n"
            f"**Résultat :** :{color}[{result_text}]  \n"
            f"**Confiance :** {prediction['confidence']:.1%}"
        )

    with right:
        st.markdown("**Probabilités par classe**")
        probs = prediction["probabilities"]
        import pandas as pd

        df = pd.DataFrame({"Classe": list(probs.keys()), "Probabilité": list(probs.values())})
        df = df.sort_values("Probabilité", ascending=False).set_index("Classe")
        st.bar_chart(df)
