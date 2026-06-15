
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
wrapped_model = wrapped_model.to('mps')
wrapped_model.eval()

print("Creating sample input...")

sequence = "MKTVRQERLKSIVRILERSKEPV"

inputs = tokenizer(
    sequence,
    return_tensors="pt"
)
inputs = {k: v.to("mps") for k, v in inputs.items()}
print("Tracing model...")

with torch.no_grad():
    traced_model = torch.jit.trace(
        wrapped_model,
        (
            inputs["input_ids"],
            inputs["attention_mask"]
        )
    )

traced_model.save("esm2_mps.pt")


loaded = torch.jit.load("esm2_mps.pt")
loaded = loaded.to("mps")

inputs = tokenizer(
    "MKTVRQERLKSIVRILERSKEPV",
    return_tensors="pt"
)

inputs = {k: v.to("mps") for k, v in inputs.items()}

with torch.no_grad():
    out = loaded(
        inputs["input_ids"],
        inputs["attention_mask"]
    )

print(out.device)
