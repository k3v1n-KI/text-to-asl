import os
import json
import torch
import torch.nn as nn
import numpy as np


class TextToLandmarkLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextToLandmarkLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, seq_len):
        emb = self.embedding(x)  # x: [batch_size]
        emb = emb.unsqueeze(1).repeat(1, seq_len, 1)  # repeat for sequence
        out, _ = self.lstm(emb)
        out = self.fc(out)  # [batch, seq_len, output_dim]
        return out

class TexttoMPPoints:
    def __init__(self):
        save_dir = os.path.join("models", "model_2")
        model_path = os.path.join(save_dir, "asl_lstm_state_dict.pth")
        vocab_path = os.path.join("models", "model_2", "vocab.json")
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        vocab_size = len(self.vocab)
        EMBED_DIM = 32
        HID_DIM   = 128
        # MediaPipe Holistic landmarks: pose 33, face 468, left_hand 21, right_hand 21
        OUTPUT_DIM = (33 + 468 + 21 + 21) * 3  

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model_loaded = TextToLandmarkLSTM(vocab_size, EMBED_DIM, HID_DIM, OUTPUT_DIM)
        self.model_loaded.load_state_dict(torch.load(model_path))
        self.model_loaded.to(self.device)
        self.model_loaded.eval()

    def predict_landmarks(self, word, seq_len=30):
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocab")
        idx = torch.tensor([self.vocab[word]], device=self.device)
        with torch.no_grad():
            preds = self.model_loaded(idx, seq_len)           # → [1, seq_len, OUTPUT_DIM]
            preds_squeeze = preds.squeeze(0).cpu().tolist()
            arranged_preds = unflatten_landmarks(preds_squeeze)
        return arranged_preds
    
def unflatten_landmarks_frame(flat_vec):
    """
    Turn a flat (1629,) prediction into a dict of landmark lists like MediaPipe.
    """
    arr = np.asarray(flat_vec, dtype=float)
    groups = [("pose", 33), ("face", 468), ("left_hand", 21), ("right_hand", 21)]
    coords = 3

    result = {}
    idx = 0
    for name, count in groups:
        pts = []
        length = count * coords
        chunk = arr[idx : idx + length]
        for i in range(count):
            x, y, z = chunk[i*coords:(i+1)*coords]
            pts.append({"x": float(x), "y": float(y), "z": float(z)})
        result[name] = pts
        idx += length

    assert idx == arr.size, f"Consumed {idx} of {arr.size}"
    return result

def unflatten_landmarks(preds):
    """
    If preds is a single flat vector → returns one dict.
    If preds is a list/array of flat vectors → returns list of dicts.
    """
    arr = np.asarray(preds)
    if arr.ndim == 1:
        # single frame
        return unflatten_landmarks_frame(arr)
    elif arr.ndim == 2:
        # sequence of frames
        return [unflatten_landmarks_frame(frame) for frame in arr]
    else:
        raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")
