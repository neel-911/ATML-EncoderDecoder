import streamlit as st
import json, pickle, os, re, string
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Lab 6: Encoder-Decoder", layout="wide")
st.title("Lab 6: Encoder-Decoder for Machine Translation & Summarization")

@st.cache_resource
def load_inference_models(model_path, latent_dim=256):
    model = load_model(model_path)
    
    l_lstm = [l for l in model.layers if 'LSTM' in l.__class__.__name__]
    enc_lstm = l_lstm[0]
    dec_lstm = l_lstm[1]
    dec_emb  = [l for l in model.layers if 'Embedding' in l.__class__.__name__][1]
    dense    = [l for l in model.layers if 'Dense' in l.__class__.__name__][0]
    
    # Disable cuDNN for M1/Metal local inference single-token prediction problem
    try:
        enc_lstm.use_cudnn = False
        dec_lstm.use_cudnn = False
    except:
        pass

    encoder_model = Model(inputs=model.input[0], outputs=enc_lstm.output[1:])
    
    dec_state_h = Input(shape=(latent_dim,))
    dec_state_c = Input(shape=(latent_dim,))
    dec_inf_in  = Input(shape=(1,))
    
    x = dec_emb(dec_inf_in)
    out, h, c = dec_lstm(x, initial_state=[dec_state_h, dec_state_c])
    out = dense(out)
    
    decoder_model = Model([dec_inf_in, dec_state_h, dec_state_c], [out, h, c])
    return encoder_model, decoder_model

def clean_text(text, max_w=None):
    text = str(text).lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = re.sub(r"\s+", " ", text).split()
    if max_w: words = words[:max_w]
    return " ".join(words)

def generate_output(input_text, src_tok, tgt_tok, max_src, max_tgt, enc_model, dec_model, max_w=None):
    cleaned = clean_text(input_text, max_w)
    if not cleaned:
        return ""
    seq = src_tok.texts_to_sequences([cleaned])
    seq = pad_sequences(seq, maxlen=max_src, padding="post")
    
    h, c = enc_model.predict(seq, verbose=0)
    
    idx2word = {v: k for k, v in tgt_tok.word_index.items()}
    start_idx = tgt_tok.word_index.get("<start>", 1)
    if start_idx is None: start_idx = 1
    
    token = np.array([[start_idx]])
    result = []
    for _ in range(max_tgt):
        out, h, c = dec_model.predict([token, h, c], verbose=0)
        idx = np.argmax(out[0, -1, :])
        word = idx2word.get(idx, "")
        if word == "<end>" or word == "":
            break
        result.append(word)
        token = np.array([[idx]])
    return " ".join(result)

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
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Test Model")
        en_input = st.text_input("Enter English text:", "How are you today?")
        if os.path.exists("eng_hindi_artifacts.pkl") and os.path.exists("eng_hindi_model.keras"):
            if st.button("Translate to Hindi"):
                with st.spinner("Translating..."):
                    enc, dec = load_inference_models("eng_hindi_model.keras")
                    with open("eng_hindi_artifacts.pkl", "rb") as f:
                        a = pickle.load(f)
                    
                    pred = generate_output(
                        en_input, a["eng_tokenizer"], a["hin_tokenizer"],
                        a["max_eng_len"], a["max_hin_len"], enc, dec
                    )
                    st.success(f"**Hindi:** {pred}")
        else:
            st.warning("Model files not found. Run `python task1_eng_hindi.py` first.")

    with col2:
        st.subheader("Training Logs")
        if os.path.exists("eng_hindi_artifacts.pkl"):
            with open("eng_hindi_artifacts.pkl", "rb") as f:
                a = pickle.load(f)
            st.pyplot(plot_history(a["history"]))
            st.info(f"Eng vocab: {len(a['eng_tokenizer'].word_index)} · Hin vocab: {len(a['hin_tokenizer'].word_index)}")
        elif os.path.exists("task1_training_plot.png"):
            st.image("task1_training_plot.png")
        else:
            st.write("No logs available.")

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
    
    c31, c32 = st.columns([1, 1])
    with c31:
        st.subheader("Test Model")
        en_input_es = st.text_input("Enter English text:", "Science expands human understanding.")
        if os.path.exists("eng_spanish_artifacts.pkl") and os.path.exists("eng_spanish_model.keras"):
            if st.button("Translate to Spanish"):
                with st.spinner("Translating..."):
                    enc3, dec3 = load_inference_models("eng_spanish_model.keras")
                    with open("eng_spanish_artifacts.pkl", "rb") as f:
                        a3 = pickle.load(f)
                    
                    pred3 = generate_output(
                        en_input_es, a3["eng_tok"], a3["spa_tok"],
                        a3["max_eng"], a3["max_spa"], enc3, dec3
                    )
                    st.success(f"**Spanish:** {pred3}")
        else:
            st.warning("Model files not found. Run `python task3_eng_spanish.py` first.")

    with c32:
        st.subheader("Training Logs")
        if os.path.exists("eng_spanish_artifacts.pkl"):
            with open("eng_spanish_artifacts.pkl", "rb") as f:
                a3 = pickle.load(f)
            st.pyplot(plot_history(a3["history"]))
        elif os.path.exists("task3_training_plot.png"):
            st.image("task3_training_plot.png")
        else:
            st.write("No logs available")

with tab4:
    st.header("Text Summarization (XSum)")
    st.markdown("**Dataset:** EdinburghNLP/xsum · 6K samples · 20 epochs")
    
    c41, c42 = st.columns([1, 1])
    with c41:
        st.subheader("Test Model")
        long_text = st.text_area(
            "Enter a paragraph (will be truncated to 40 words):",
            "Transformers changed NLP by enabling better parallelization and context handling. "
            "Encoder-decoder variants are especially useful for generation tasks where input and output "
            "sequences differ, such as translation and summarization."
        )
        if os.path.exists("summarization_artifacts.pkl") and os.path.exists("summarization_model.keras"):
            if st.button("Generate Summary"):
                with st.spinner("Summarizing..."):
                    enc4, dec4 = load_inference_models("summarization_model.keras")
                    with open("summarization_artifacts.pkl", "rb") as f:
                        a4 = pickle.load(f)
                    
                    # Assuming task4 hardcoded lengths: MAX_SRC_LEN=40, max_tgt=16
                    pred4 = generate_output(
                        long_text, a4["src_tok"], a4["tgt_tok"],
                        40, 16, enc4, dec4, max_w=40
                    )
                    st.success(f"**Summary:** {pred4}")
        else:
            st.warning("Model files not found. Run `python task4_summarization.py` first.")

    with c42:
        st.subheader("Training Logs")
        if os.path.exists("summarization_artifacts.pkl"):
            with open("summarization_artifacts.pkl", "rb") as f:
                a4 = pickle.load(f)
            st.pyplot(plot_history(a4["history"]))
        elif os.path.exists("task4_training_plot.png"):
            st.image("task4_training_plot.png")
        else:
            st.write("No logs available.")