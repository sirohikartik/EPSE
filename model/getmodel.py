"""Script to download and TorchScript-trace ESM-2 on CPU."""

import argparse
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "facebook/esm2_t12_35M_UR50D"


class ESMWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state


def main():
    parser = argparse.ArgumentParser(description="Export traced ESM-2 model on CPU")
    parser.add_argument("--output", "-o", default="esm2_cpu.pt", help="Path to save TorchScript model")
    parser.add_argument("--device", "-d", default="cpu", help="Device to trace on (defaults to cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading model on {device}...")
    base_model = AutoModel.from_pretrained(MODEL_NAME)
    base_model.eval()

    wrapped_model = ESMWrapper(base_model).to(device)
    wrapped_model.eval()

    sample_sequence = "MKTVRQERLKSIVRILERSKEPV"
    inputs = tokenizer(sample_sequence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("Tracing model with TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(
            wrapped_model,
            (
                inputs["input_ids"],
                inputs["attention_mask"],
            ),
        )

    traced_model.save(args.output)
    print(f"Successfully saved traced CPU model to {args.output}")


if __name__ == "__main__":
    main()
