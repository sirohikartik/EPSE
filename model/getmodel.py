from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

print("Creating wrapper...")

class ESMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state

wrapped_model = ESMWrapper(model)
wrapped_model.eval()

print("Creating sample input...")

sequence = "MKTVRQERLKSIVRILERSKEPV"

inputs = tokenizer(
    sequence,
    return_tensors="pt"
)

print("Tracing model...")

with torch.no_grad():
    traced_model = torch.jit.trace(
        wrapped_model,
        (
            inputs["input_ids"],
            inputs["attention_mask"]
        )
    )

print("Saving TorchScript model...")

traced_model.save("esm2_t12_35M.pt")

print("Saved: esm2_t12_35M.pt")

print("Saving tokenizer...")
tokenizer.save_pretrained("./esm2_t12_35M_tokenizer")
