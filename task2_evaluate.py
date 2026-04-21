import numpy as np
import pickle
import json
import re, string
import pandas as pd
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)

# --- Load ---
model = load_model("eng_hindi_model.keras")
with open("eng_hindi_artifacts.pkl", "rb") as f:
    art = pickle.load(f)

eng_tok   = art["eng_tokenizer"]
hin_tok   = art["hin_tokenizer"]
max_eng   = art["max_eng_len"]
max_hin   = art["max_hin_len"]
LATENT_DIM = 256

hin_idx2word = {v: k for k, v in hin_tok.word_index.items()}

# ============================================================
# FIX: Disable cuDNN on both LSTM layers so Metal doesn't
#      choke on single-token masked inputs during inference
# ============================================================
model.get_layer("encoder_lstm").use_cudnn = False
model.get_layer("decoder_lstm").use_cudnn = False

# --- Encoder inference model ---
encoder_model = Model(
    inputs=model.input[0],
    outputs=model.get_layer("encoder_lstm").output[1:]  # [h, c]
)

# --- Decoder inference model ---
dec_state_h = Input(shape=(LATENT_DIM,), name="inf_h")
dec_state_c = Input(shape=(LATENT_DIM,), name="inf_c")
dec_inf_in  = Input(shape=(1,), name="inf_dec_input")

dec_emb_layer = model.get_layer("decoder_embedding")
dec_lstm      = model.get_layer("decoder_lstm")
dec_dense     = model.get_layer("dense_output")

dec_emb_inf = dec_emb_layer(dec_inf_in)
dec_out_inf, h_inf, c_inf = dec_lstm(dec_emb_inf, initial_state=[dec_state_h, dec_state_c])
dec_out_inf = dec_dense(dec_out_inf)

decoder_model = Model(
    [dec_inf_in, dec_state_h, dec_state_c],
    [dec_out_inf, h_inf, c_inf],
)

# --- Decode function ---
def decode_sequence(input_text):
    seq = eng_tok.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_eng, padding="post")
    h, c = encoder_model.predict(seq, verbose=0)

    token = np.array([[hin_tok.word_index.get("<start>", 1)]])
    result = []
    for _ in range(max_hin):
        out, h, c = decoder_model.predict([token, h, c], verbose=0)
        idx = np.argmax(out[0, -1, :])
        word = hin_idx2word.get(idx, "")
        if word == "<end>" or word == "":
            break
        result.append(word)
        token = np.array([[idx]])
    return " ".join(result)

# --- Evaluate ---
def clean(text):
    text = str(text).lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return re.sub(r"\s+", " ", text).strip()

df = pd.read_csv("Hindi_English_Truncated_Corpus.csv")
df = df[["english_sentence", "hindi_sentence"]].dropna()
test_df = df.iloc[8000:8100]  # sentences NOT seen during training

refs, hyps, samples = [], [], []
smooth = SmoothingFunction().method1

print("Translating test sentences...")
for i, (_, row) in enumerate(test_df.iterrows()):
    src  = clean(row["english_sentence"])
    ref  = clean(row["hindi_sentence"])
    pred = decode_sequence(src)

    refs.append([ref.split()])
    hyps.append(pred.split())
    samples.append({"english": src, "reference_hindi": ref, "predicted_hindi": pred})
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/100 done...")

bleu = corpus_bleu(refs, hyps, smoothing_function=smooth)
print(f"\nCorpus BLEU: {bleu:.4f}\n")
print(f"{'English':<35} {'Reference':<35} {'Predicted':<35}")
print("-" * 105)
for s in samples[:10]:
    print(f"{s['english'][:34]:<35} {s['reference_hindi'][:34]:<35} {s['predicted_hindi'][:34]:<35}")

with open("task2_results.json", "w") as f:
    json.dump({"bleu_score": bleu, "samples": samples[:20]}, f, ensure_ascii=False, indent=2)

print("\nTask 2 complete!")