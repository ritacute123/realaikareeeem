
import torch
from model import AceAssistantModel
from tokenizer import AceTokenizer
import pickle

# Load tokenizer
with open("ace_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load model
model = AceAssistantModel(vocab_size=30522)
model.load_state_dict(torch.load("ace_assistant_model.pt", map_location=torch.device("cpu")))
model.eval()

def run_inference(text):
    input_ids = tokenizer.encode(text)
    input_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_ids = torch.argmax(logits, dim=-1).squeeze().tolist()
    decoded = tokenizer.decode(predicted_ids)
    return decoded
