import numpy as np
import pickle
import re, string
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# ===================== CONFIG =====================
SAMPLES    = 8000
MAX_LEN    = 15
EMB_DIM    = 128
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS     = 25
# ==================================================

# --- Load opus_books ---
print("Downloading opus_books en-es...")
ds = load_dataset("opus_books", "en-es", split="train")

def clean(text):
    text = str(text).lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return re.sub(r"\s+", " ", text).strip()

pairs = []
for item in ds:
    e = clean(item["translation"]["en"])
    s = clean(item["translation"]["es"])
    if 2 <= len(e.split()) <= MAX_LEN and 2 <= len(s.split()) <= MAX_LEN:
        pairs.append((e, s))
    if len(pairs) >= SAMPLES:
        break

print(f"Using {len(pairs)} pairs")
eng_texts    = [p[0] for p in pairs]
spa_texts_in = ["<start> " + p[1] for p in pairs]
spa_texts_tgt= [p[1] + " <end>" for p in pairs]

# --- Tokenize ---
eng_tok = Tokenizer(filters=""); eng_tok.fit_on_texts(eng_texts)
spa_tok = Tokenizer(filters=""); spa_tok.fit_on_texts(spa_texts_in + spa_texts_tgt)

eng_vocab = len(eng_tok.word_index) + 1
spa_vocab = len(spa_tok.word_index) + 1
max_eng = MAX_LEN
max_spa = MAX_LEN + 1

enc_input  = pad_sequences(eng_tok.texts_to_sequences(eng_texts), maxlen=max_eng, padding="post")
dec_input  = pad_sequences(spa_tok.texts_to_sequences(spa_texts_in), maxlen=max_spa, padding="post")
dec_target = pad_sequences(spa_tok.texts_to_sequences(spa_texts_tgt), maxlen=max_spa, padding="post")

enc_tr, enc_te, din_tr, din_te, dtgt_tr, dtgt_te = train_test_split(
    enc_input, dec_input, dec_target, test_size=0.2, random_state=42
)
dtgt_tr = np.expand_dims(dtgt_tr, -1)
dtgt_te = np.expand_dims(dtgt_te, -1)

# --- Model ---
ei = Input(shape=(max_eng,))
ee = Embedding(eng_vocab, EMB_DIM, mask_zero=True)(ei)
_, h, c = LSTM(LATENT_DIM, return_state=True)(ee)

di = Input(shape=(max_spa,))
de = Embedding(spa_vocab, EMB_DIM, mask_zero=True)(di)
do, _, _ = LSTM(LATENT_DIM, return_sequences=True, return_state=True)(de, initial_state=[h, c])
do = Dense(spa_vocab, activation="softmax")(do)

model = Model([ei, di], do)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    [enc_tr, din_tr], dtgt_tr,
    batch_size=BATCH_SIZE, epochs=EPOCHS,
    validation_data=([enc_te, din_te], dtgt_te),
)

model.save("eng_spanish_model.keras")
with open("eng_spanish_artifacts.pkl", "wb") as f:
    pickle.dump({
        "eng_tok": eng_tok, "spa_tok": spa_tok,
        "max_eng": max_eng, "max_spa": max_spa,
        "history": history.history,
    }, f)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(history.history["loss"], label="Train"); a1.plot(history.history["val_loss"], label="Val")
a1.set_title("Loss"); a1.legend()
a2.plot(history.history["accuracy"], label="Train"); a2.plot(history.history["val_accuracy"], label="Val")
a2.set_title("Accuracy"); a2.legend()
fig.tight_layout(); fig.savefig("task3_training_plot.png", dpi=150)
print("Task 3 complete!")