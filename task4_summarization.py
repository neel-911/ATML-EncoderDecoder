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
SAMPLES     = 6000
MAX_SRC_LEN = 40       # article truncated to 40 words
MAX_TGT_LEN = 15       # summary truncated to 15 words
EMB_DIM     = 128
LATENT_DIM  = 256
BATCH_SIZE  = 64
EPOCHS      = 20
VOCAB_LIMIT = 12000
# ==================================================

print("Downloading XSum dataset...")
ds = load_dataset("EdinburghNLP/xsum", split="train")
data = ds.select(range(SAMPLES))

def clean(text, max_w):
    text = str(text).lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return " ".join(re.sub(r"\s+", " ", text).split()[:max_w])

src = [clean(item["document"], MAX_SRC_LEN) for item in data]
tgt_in  = ["<start> " + clean(item["summary"], MAX_TGT_LEN) for item in data]
tgt_out = [clean(item["summary"], MAX_TGT_LEN) + " <end>" for item in data]

src_tok = Tokenizer(num_words=VOCAB_LIMIT, filters=""); src_tok.fit_on_texts(src)
tgt_tok = Tokenizer(num_words=VOCAB_LIMIT, filters=""); tgt_tok.fit_on_texts(tgt_in + tgt_out)

src_vocab = min(VOCAB_LIMIT, len(src_tok.word_index) + 1)
tgt_vocab = min(VOCAB_LIMIT, len(tgt_tok.word_index) + 1)
max_tgt = MAX_TGT_LEN + 1

src_seq = pad_sequences(src_tok.texts_to_sequences(src), maxlen=MAX_SRC_LEN, padding="post")
din_seq = pad_sequences(tgt_tok.texts_to_sequences(tgt_in), maxlen=max_tgt, padding="post")
dout_seq = pad_sequences(tgt_tok.texts_to_sequences(tgt_out), maxlen=max_tgt, padding="post")

s_tr, s_te, di_tr, di_te, dt_tr, dt_te = train_test_split(
    src_seq, din_seq, dout_seq, test_size=0.2, random_state=42
)
dt_tr = np.expand_dims(dt_tr, -1)
dt_te = np.expand_dims(dt_te, -1)

ei = Input(shape=(MAX_SRC_LEN,))
ee = Embedding(src_vocab, EMB_DIM, mask_zero=True)(ei)
_, h, c = LSTM(LATENT_DIM, return_state=True)(ee)

di = Input(shape=(max_tgt,))
de = Embedding(tgt_vocab, EMB_DIM, mask_zero=True)(di)
do, _, _ = LSTM(LATENT_DIM, return_sequences=True, return_state=True)(de, initial_state=[h, c])
do = Dense(tgt_vocab, activation="softmax")(do)

model = Model([ei, di], do)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit([s_tr, di_tr], dt_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,
                    validation_data=([s_te, di_te], dt_te))

model.save("summarization_model.keras")
with open("summarization_artifacts.pkl", "wb") as f:
    pickle.dump({"src_tok": src_tok, "tgt_tok": tgt_tok, "history": history.history}, f)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(history.history["loss"], label="Train"); a1.plot(history.history["val_loss"], label="Val")
a1.set_title("Loss"); a1.legend()
a2.plot(history.history["accuracy"], label="Train"); a2.plot(history.history["val_accuracy"], label="Val")
a2.set_title("Accuracy"); a2.legend()
fig.tight_layout(); fig.savefig("task4_training_plot.png", dpi=150)
print("Task 4 complete!")