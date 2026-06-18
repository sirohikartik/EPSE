import os
import sys
import torch
import numpy as np
import csv
import struct
from transformers import AutoTokenizer

# Configuration
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
DIM = 480
DEFAULT_THRESHOLD = 0.90
DEFAULT_FASTA = "uniprot_sprot.fasta"
BATCH_SIZE = 8

def get_embeddings_batch(model, tokenizer, sequences, device):
    inputs = tokenizer(
        sequences, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
        
        if isinstance(outputs, torch.Tensor):
            hidden_states = outputs
        elif isinstance(outputs, (tuple, list)):
            hidden_states = outputs[0]
        elif isinstance(outputs, dict):
            hidden_states = outputs.get("last_hidden_state")
        else:
            try:
                hidden_states = outputs.last_hidden_state
            except AttributeError:
                hidden_states = outputs

        embeddings = torch.mean(hidden_states, dim=1)
        
    return embeddings.cpu().numpy()

def cosine_similarity(centroid_vec, embedding_vec):
    dot_product = np.dot(centroid_vec, embedding_vec)
    norm_c = np.linalg.norm(centroid_vec)
    norm_e = np.linalg.norm(embedding_vec)
    if norm_c == 0 or norm_e == 0:
        return 0.0
    return dot_product / (norm_c * norm_e)

def update_centroid(centroid_data, embedding_vec):
    centroid_data['count'] += 1
    centroid_data['vec'] += (embedding_vec - centroid_data['vec']) / centroid_data['count']

def parse_fasta(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: File {filepath} not found.")
        return

    with open(filepath, 'r') as f:
        entry = None
        sequence = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if entry:
                    entry['seq'] = "".join(sequence)
                    yield entry
                
                header = line[1:]
                parts = header.split('|')
                accession = parts[1] if len(parts) > 1 else "unknown"
                
                remaining = parts[2] if len(parts) > 2 else ""
                if ' ' in remaining:
                    entry_name, desc_part = remaining.split(' ', 1)
                else:
                    entry_name, desc_part = remaining, ""
                
                organism = "unknown"
                if "OS=" in desc_part:
                    org_start = desc_part.find("OS=") + 3
                    org_end = desc_part.find(" OX=", org_start)
                    if org_end == -1:
                        organism = desc_part[org_start:].strip()
                    else:
                        organism = desc_part[org_start:org_end].strip()
                
                description = desc_part.split(" OS=")[0].strip()

                entry = {
                    'accession': accession,
                    'entry_name': entry_name,
                    'description': description,
                    'organism': organism
                }
                sequence = []
            else:
                sequence.append(line)
        
        if entry:
            entry['seq'] = "".join(sequence)
            yield entry

def parse_and_save(fasta_path, threshold, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    
    proteins_path = os.path.join(data_dir, "proteins.csv")
    emb_path = os.path.join(data_dir, "embeddings.bin")
    assign_path = os.path.join(data_dir, "assignments.bin")
    cent_path = os.path.join(data_dir, "centroids.csv")

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    
    try:
        model = torch.jit.load("esm2_mps.pt").to(device)
        print("Loaded JIT model.")
    except Exception as e:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        print(f"Loaded Transformers model (JIT load failed: {e})")

    centroids = [] 
    
    print("Parsing started...")
    # Open files in 'a' (append) or 'w' mode. Since we want to save continuously, 
    # we use 'w' for the start of a fresh run but we will flush periodically.
    with open(proteins_path, 'w', newline='') as csv_f, \
         open(emb_path, 'ab') as emb_f, \
         open(assign_path, 'ab') as assign_f:
        
        # Note: emb_path and assign_path opened in 'ab' but they should probably start empty
        # Let's truncate them first if we are starting from scratch
        emb_f.truncate(0)
        assign_f.truncate(0)

        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(["id", "accession", "entry_name", "description", "organism", "seq_len"])
        
        protein_id = 0
        batch_entries = []
        batch_sequences = []
        
        for entry in parse_fasta(fasta_path):
            batch_entries.append(entry)
            batch_sequences.append(entry['seq'])
            
            if len(batch_sequences) >= BATCH_SIZE:
                embeddings = get_embeddings_batch(model, tokenizer, batch_sequences, device)
                
                for i, embedding in enumerate(embeddings):
                    protein_id += 1
                    e = batch_entries[i]
                    
                    emb_f.write(struct.pack('I', protein_id))
                    emb_f.write(embedding.astype(np.float32).tobytes())
                    
                    best_score = -2.0
                    best_centroid_idx = -1
                    for idx, c in enumerate(centroids):
                        score = cosine_similarity(c['vec'], embedding)
                        if score > best_score:
                            best_score = score
                            best_centroid_idx = idx
                    
                    if best_centroid_idx != -1 and best_score >= threshold:
                        cluster_id = centroids[best_centroid_idx]['id']
                        update_centroid(centroids[best_centroid_idx], embedding)
                    else:
                        cluster_id = len(centroids)
                        centroids.append({
                            'id': cluster_id,
                            'count': 1,
                            'vec': embedding.copy()
                        })
                    
                    assign_f.write(struct.pack('II', protein_id, cluster_id))
                    csv_writer.writerow([
                        protein_id, e['accession'], e['entry_name'], 
                        e['description'], e['organism'], len(e['seq'])
                    ])
                    
                    if protein_id % 10 == 0:
                        print(f"[{protein_id} done]")
                        # Force write to disk
                        csv_f.flush()
                        emb_f.flush()
                        assign_f.flush()
                
                batch_entries = []
                batch_sequences = []

        if batch_sequences:
            embeddings = get_embeddings_batch(model, tokenizer, batch_sequences, device)
            for i, embedding in enumerate(embeddings):
                protein_id += 1
                e = batch_entries[i]
                emb_f.write(struct.pack('I', protein_id))
                emb_f.write(embedding.astype(np.float32).tobytes())
                
                best_score = -2.0
                best_centroid_idx = -1
                for idx, c in enumerate(centroids):
                    score = cosine_similarity(c['vec'], embedding)
                    if score > best_score:
                        best_score = score
                        best_centroid_idx = idx
                
                if best_centroid_idx != -1 and best_score >= threshold:
                    cluster_id = centroids[best_centroid_idx]['id']
                    update_centroid(centroids[best_centroid_idx], embedding)
                else:
                    cluster_id = len(centroids)
                    centroids.append({'id': cluster_id, 'count': 1, 'vec': embedding.copy()})
                
                assign_f.write(struct.pack('II', protein_id, cluster_id))
                csv_writer.writerow([
                    protein_id, e['accession'], e['entry_name'], 
                    e['description'], e['organism'], len(e['seq'])
                ])
                if protein_id % 10 == 0:
                    print(f"[{protein_id} done]")

    with open(cent_path, 'w', newline='') as cent_f:
        cent_writer = csv.writer(cent_f)
        header = ["centroid_id", "count"] + [f"e{i}" for i in range(DIM)]
        cent_writer.writerow(header)
        for c in centroids:
            row = [c['id'], c['count']] + c['vec'].tolist()
            cent_writer.writerow(row)

def main():
    threshold = DEFAULT_THRESHOLD
    fasta_path = DEFAULT_FASTA

    if len(sys.argv) >= 2:
        try:
            threshold = float(sys.argv[1])
        except ValueError:
            print("ERROR: threshold must be a float.")
            sys.exit(1)
    
    if len(sys.argv) >= 3:
        fasta_path = sys.argv[2]

    print("=== EPSE Clustering (Python Optimized) ===")
    print(f"Threshold : {threshold:.2f}")
    print(f"Dataset   : {fasta_path}\n")

    data_dir = f".data/threshold_{threshold:.2f}"
    parse_and_save(fasta_path, threshold, data_dir)

if __name__ == "__main__":
    main()
