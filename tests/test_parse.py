import os
import shutil
import pytest
import torch
import numpy as np
from parser.parse import parse_and_save, parse_fasta

def test_parse_fasta():
    # Create a dummy fasta file
    test_fasta = "test_dummy.fasta"
    with open(test_fasta, "w") as f:
        f.write(">db|prot1|name1 description1 OS=Org1 OX=123\nSEQ1\n")
        f.write(">db|prot2|name2 description2 OS=Org2 OX=456\nSEQ2\n")
    
    try:
        entries = list(parse_fasta(test_fasta))
        assert len(entries) == 2
        assert entries[0]['accession'] == 'prot1'
        assert entries[0]['organism'] == 'Org1'
        assert entries[0]['seq'] == 'SEQ1'
        assert entries[1]['accession'] == 'prot2'
    finally:
        if os.path.exists(test_fasta):
            os.remove(test_fasta)

def test_parse_and_save_flow(tmp_path):
    # Setup a small fasta file for integration testing
    test_fasta = tmp_path / "test_integration.fasta"
    test_fasta.write_text(">db|p1|n1 d1 OS=O1 OX=1\nMKTV\n")
    
    # We use a very low threshold to ensure clustering works as expected
    threshold = 0.1
    data_dir = str(tmp_path / "output_data")
    
    # Since parse_and_save loads a large model, we might want to mock the model
    # but for a basic integration test, we'll check if it creates the files.
    # Note: This test might be slow because it loads the ESM2 model.
    try:
        parse_and_save(str(test_fasta), threshold, data_dir)
        
        assert os.path.exists(os.path.join(data_dir, "proteins.csv"))
        assert os.path.exists(os.path.join(data_dir, "embeddings.bin"))
        assert os.path.exists(os.path.join(data_dir, "assignments.bin"))
        assert os.path.exists(os.path.join(data_dir, "centroids.csv"))
    except Exception as e:
        pytest.fail(f"parse_and_save failed: {e}")

if __name__ == "__main__":
    # Allow running directly for quick check
    pytest.main([__file__])
EOF"
