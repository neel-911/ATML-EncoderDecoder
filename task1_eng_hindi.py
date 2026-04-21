import numpy as np
import pandas as pd
import pickle
import re, string
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ===================== CONFIG =====================
SAMPLES    = 16000      # slight bump from 8K
MAX_LEN    = 15         # same as before
EMB_DIM    = 128        # same
LATENT_DIM = 256        # same
DROPOUT    = 0.2        # light dropout — biggest bang for buck
BATCH_SIZE = 64
EPOCHS     = 40         # more room, EarlyStopping handles the rest
PATIENCE   = 5
# ==================================================

# --- Load data ---
df = pd.read_csv("Hindi_English_Truncated_Corpus.csv")
df = df[["english_sentence", "hindi_sentence"]].dropna()

def clean(text):
    text = str(text).lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return re.sub(r"\s+", " ", text).strip()

df["eng_clean"] = df["english_sentence"].apply(clean)
df["hin_clean"] = df["hindi_sentence"].apply(clean)
df = df[df["eng_clean"].apply(lambda x: 2 <= len(x.split()) <= MAX_LEN)]
df = df[df["hin_clean"].apply(lambda x: 2 <= len(x.split()) <= MAX_LEN)]
df = df.head(SAMPLES)
print(f"Using {len(df)} sentence pairs")

eng_texts      = df["eng_clean"].tolist()
hin_texts_in   = ["<start> " + t for t in df["hin_clean"].tolist()]
hin_texts_tgt  = [t + " <end>" for t in df["hin_clean"].tolist()]

# --- Tokenize ---
eng_tok = Tokenizer(filters=""); eng_tok.fit_on_texts(eng_texts)
hin_tok = Tokenizer(filters=""); hin_tok.fit_on_texts(hin_texts_in + hin_texts_tgt)

eng_vocab = len(eng_tok.word_index) + 1
hin_vocab = len(hin_tok.word_index) + 1
max_eng   = MAX_LEN
max_hin   = MAX_LEN + 1

print(f"Eng vocab: {eng_vocab}, Hin vocab: {hin_vocab}")

enc_input  = pad_sequences(eng_tok.texts_to_sequences(eng_texts), maxlen=max_eng, padding="post")
dec_input  = pad_sequences(hin_tok.texts_to_sequences(hin_texts_in), maxlen=max_hin, padding="post")
dec_target = pad_sequences(hin_tok.texts_to_sequences(hin_texts_tgt), maxlen=max_hin, padding="post")

# --- Split ---
enc_tr, enc_te, din_tr, din_te, dtgt_tr, dtgt_te = train_test_split(
    enc_input, dec_input, dec_target, test_size=0.2, random_state=42
)
dtgt_tr = np.expand_dims(dtgt_tr, -1)
dtgt_te = np.expand_dims(dtgt_te, -1)

# --- Model (same size, just added dropout) ---
encoder_inputs = Input(shape=(max_eng,), name="encoder_input")
enc_emb = Embedding(eng_vocab, EMB_DIM, mask_zero=True)(encoder_inputs)
enc_emb = Dropout(DROPOUT)(enc_emb)
_, state_h, state_c = LSTM(LATENT_DIM, return_state=True, dropout=DROPOUT, name="encoder_lstm")(enc_emb)

decoder_inputs = Input(shape=(max_hin,), name="decoder_input")
dec_emb = Embedding(hin_vocab, EMB_DIM, mask_zero=True, name="decoder_embedding")(decoder_inputs)
dec_emb = Dropout(DROPOUT)(dec_emb)
dec_out, _, _ = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=DROPOUT, name="decoder_lstm")(
    dec_emb, initial_state=[state_h, state_c]
)
dec_out = Dense(hin_vocab, activation="softmax", name="dense_output")(dec_out)

model = Model([encoder_inputs, decoder_inputs], dec_out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# --- Train with EarlyStopping ---
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    [enc_tr, din_tr], dtgt_tr,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=([enc_te, din_te], dtgt_te),
    callbacks=[early_stop],
)

# --- Save ---
model.save("eng_hindi_model.keras")
with open("eng_hindi_artifacts.pkl", "wb") as f:
    pickle.dump({
        "eng_tokenizer": eng_tok,
        "hin_tokenizer": hin_tok,
        "max_eng_len": max_eng,
        "max_hin_len": max_hin,
        "history": history.history,
    }, f)

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history.history["loss"], label="Train")
ax1.plot(history.history["val_loss"], label="Val")
ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
ax2.plot(history.history["accuracy"], label="Train")
ax2.plot(history.history["val_accuracy"], label="Val")
ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
fig.tight_layout(); fig.savefig("task1_training_plot.png", dpi=150)
print(f"\nTask 1 complete! Stopped at epoch {len(history.history['loss'])}/{EPOCHS}")