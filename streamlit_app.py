import streamlit as st
import json, pickle, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Lab 6: Encoder-Decoder", layout="wide")
st.title("Lab 6: Encoder-Decoder for Machine Translation & Summarization")

def plot_history(history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(history["loss"], label="Train"); a1.plot(history["val_loss"], label="Val")
    a1.set_title("Loss"); a1.set_xlabel("Epoch"); a1.legend()
    a2.plot(history["accuracy"], label="Train"); a2.plot(history["val_accuracy"], label="Val")
    a2.set_title("Accuracy"); a2.set_xlabel("Epoch"); a2.legend()
    fig.tight_layout()
    return fig

tab1, tab2, tab3, tab4 = st.tabs([
    "Task 1: Eng→Hindi", "Task 2: Evaluation", "Task 3: Eng→Spanish", "Task 4: Summarization"
])

with tab1:
    st.header("English → Hindi Translation")
    st.markdown("**Arch:** Embedding → LSTM Encoder → hidden states → LSTM Decoder → Dense Softmax")
    st.markdown("**Dataset:** Hindi-English Truncated Corpus (Kaggle) · 8K pairs · 25 epochs")
    if os.path.exists("eng_hindi_artifacts.pkl"):
        with open("eng_hindi_artifacts.pkl", "rb") as f:
            a = pickle.load(f)
        st.pyplot(plot_history(a["history"]))
        st.info(f"Eng vocab: {len(a['eng_tokenizer'].word_index)} · Hin vocab: {len(a['hin_tokenizer'].word_index)}")
    elif os.path.exists("task1_training_plot.png"):
        st.image("task1_training_plot.png")
    else:
        st.warning("Run `python task1_eng_hindi.py` first")

with tab2:
    st.header("Evaluation & BLEU Score")
    if os.path.exists("task2_results.json"):
        with open("task2_results.json") as f:
            r = json.load(f)
        st.metric("Corpus BLEU", f"{r['bleu_score']:.4f}")
        st.subheader("Sample Translations")
        for s in r["samples"]:
            c1, c2, c3 = st.columns(3)
            c1.write(f"**Eng:** {s['english']}")
            c2.write(f"**Ref:** {s['reference_hindi']}")
            c3.write(f"**Pred:** {s['predicted_hindi']}")
            st.divider()
    else:
        st.warning("Run `python task2_evaluate.py` first")

with tab3:
    st.header("English → Spanish (opus_books)")
    st.markdown("**Dataset:** HuggingFace `opus_books` en-es · 8K pairs · 25 epochs")
    if os.path.exists("eng_spanish_artifacts.pkl"):
        with open("eng_spanish_artifacts.pkl", "rb") as f:
            a3 = pickle.load(f)
        st.pyplot(plot_history(a3["history"]))
    elif os.path.exists("task3_training_plot.png"):
        st.image("task3_training_plot.png")
    else:
        st.warning("Run `python task3_eng_spanish.py` first")

with tab4:
    st.header("Text Summarization (XSum)")
    st.markdown("**Dataset:** EdinburghNLP/xsum · 6K samples · 20 epochs")
    if os.path.exists("summarization_artifacts.pkl"):
        with open("summarization_artifacts.pkl", "rb") as f:
            a4 = pickle.load(f)
        st.pyplot(plot_history(a4["history"]))
    elif os.path.exists("task4_training_plot.png"):
        st.image("task4_training_plot.png")
    else:
        st.warning("Run `python task4_summarization.py` first")