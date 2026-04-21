# Encoder-Decoder Sequence-to-Sequence Models

A comprehensive implementation of sequence-to-sequence (Seq2Seq) models that I built for my Advanced Topics in ML course using LSTM-based Encoder-Decoder architectures for Machine Translation and Text Summarization.

##  Project Overview
This repository contains four main tasks exploring different applications of Encoder-Decoder models:
- **Task 1:** English to Hindi translation using the Hindi-English Truncated Corpus.
- **Task 2:** Evaluation of Task 1 using BLEU scores and qualitative analysis.
- **Task 3:** English to Spanish translation using the `opus_books` dataset.
- **Task 4:** Abstractive Text Summarization using the `XSum` dataset.

##  File Structure
- `task1_eng_hindi.py`: Training script for EN-HI translation.
- `task2_evaluate.py`: Evaluation and metrics calculation for EN-HI.
- `task3_eng_spanish.py`: Training script for EN-ES translation.
- `task4_summarization.py`: Training script for text summarization.
- `streamlit_app.py`: Interactive Streamlit dashboard to test all models.
- `requirements.txt`: Python dependencies.

##  Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
Execute the tasks sequentially to train the models and generate artifact files:
```bash
python task1_eng_hindi.py
python task2_evaluate.py
python task3_eng_spanish.py
python task4_summarization.py
```

### 3. Launch the Web Interface
Once models are trained (`.keras` and `.pkl` files are generated), run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

##  Sample Results
The models use:
- **Architecture**: LSTM-based Encoder-Decoder with Embedding layers and Softmax output.
- **Optimization**: Adam optimizer with Sparse Categorical Crossentropy loss.
- **Monitoring**: Early stopping and training/validation plots.

##  Note
Large model files (`.keras`) and dataset files (`.csv`) are ignored by Git to keep the repository lightweight. Use the training scripts to regenerate them locally.

