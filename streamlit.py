# app.py

import streamlit as st
import torch
import pickle
import numpy as np
import re
from torch import nn

# -----------------------
# 1️⃣ Load Artifacts
# -----------------------
with open("stoi.pkl", "rb") as f:
    stoi = pickle.load(f)

with open("itos.pkl", "rb") as f:
    itos = pickle.load(f)

with open("label_classes.pkl", "rb") as f:
    label_classes = pickle.load(f)

embedding_matrix = np.load("embedding_matrix.npy")
embedding_tensor = torch.tensor(embedding_matrix)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# 2️⃣ Define Model
# -----------------------
class BiderctionalLstm(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, dropout=0.2, num_classes=6):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=stoi["<pad>"])
        self.emb.weight.data.copy_(embedding_tensor)
        self.emb.weight.requires_grad = False

        self.lstm1 = nn.LSTM(emb_dim, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(128*2, 64, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(64*2, 32, batch_first=True, bidirectional=True)

        self.network = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32*2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, (hidden, cell) = self.lstm3(x)
        hidden_cat = torch.cat([hidden[0], hidden[1]], dim=1)
        out = self.network(hidden_cat)
        return out

# Initialize and load model
VOCAB_SIZE = len(itos)
model = BiderctionalLstm(VOCAB_SIZE).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# -----------------------
# 3️⃣ Preprocessing Functions
# -----------------------
PAD, UNK = "<pad>", "<unk>"
MAX_LEN = 178

def remove_punc(text):
    return re.sub(r"[^\w\s]", "", text)

def lemmatize_text_nltk(text):
    if isinstance(text, float) or not text:
        return []
    return text.lower().split()  # simple tokenization

def encode(toks, max_len=MAX_LEN):
    idxs = [stoi.get(t, stoi[UNK]) for t in toks][:max_len]
    if len(idxs) < max_len:
        idxs += [stoi[PAD]] * (max_len - len(idxs))
    return idxs

# -----------------------
# 4️⃣ Prediction Function
# -----------------------
def predict_emotion(text):
    text = remove_punc(text)
    tokens = lemmatize_text_nltk(text)
    input_ids = torch.tensor([encode(tokens)], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_ids)
        pred_idx = outputs.argmax(dim=1).item()
    
    return label_classes[pred_idx]

# -----------------------
# 5️⃣ Streamlit App UI
# -----------------------
st.title("Emotion Classifier")
st.write("Enter a sentence and get the predicted emotion.")

user_input = st.text_area("Type your sentence here:")

if st.button("Predict"):
    if user_input.strip() != "":
        emotion = predict_emotion(user_input)
        st.success(f"Predicted Emotion: {emotion}")
    else:
        st.warning("Please enter a sentence.")
