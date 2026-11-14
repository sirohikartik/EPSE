"""Tests for EPSE command-line interface."""

import os
from epse.cli import build_parser


def test_cli_index_and_search_integration(tmp_path):
    parser = build_parser()

    # Create small fasta
    fasta_file = tmp_path / "sample.fasta"
    fasta_file.write_text(
        ">sp|P11111|TEST1 Protein One OS=Homo sapiens OX=9606 GN=T1 PE=1 SV=1\n"
        "MALWMRLLPLLALLALWGPDPAAA\n"
        ">sp|P22222|TEST2 Protein Two OS=Mus musculus OX=10090 GN=T2 PE=1 SV=1\n"
        "MKTVRQERLKSIVRILERSKEPV\n"
    )

    data_dir = str(tmp_path / "cli_index")

    # Run index command via parser
    args_index = parser.parse_args([
        "index",
        str(fasta_file),
        "--data-dir", data_dir,
        "--k", "2",
        "--use-mock",
    ])
    code_index = args_index.func(args_index)
    assert code_index == 0

    assert os.path.exists(os.path.join(data_dir, "proteins.csv"))
    assert os.path.exists(os.path.join(data_dir, "centroids.csv"))

    # Run search command with sequence and quantization flag
    args_search = parser.parse_args([
        "search",
        "MALWMRLLPLLALLALWGPDPAAA",
        "--data-dir", data_dir,
        "--top-k", "1",
        "--use-mock",
        "--quantize",
        "--format", "json",
    ])
    code_search = args_search.func(args_search)
    assert code_search == 0

    # Run search command with .npy vector query (pre-embedded mode)
    import numpy as np
    query_npy = tmp_path / "query.npy"
    np.save(str(query_npy), np.random.randn(2, 480).astype(np.float32))
    args_search_vec = parser.parse_args([
        "search",
        str(query_npy),
        "--data-dir", data_dir,
        "--top-k", "1",
        "--format", "json",
    ])
    code_search_vec = args_search_vec.func(args_search_vec)
    assert code_search_vec == 0

    # Run info command
    args_info = parser.parse_args([
        "info",
        "--data-dir", data_dir,
    ])
    code_info = args_info.func(args_info)
    assert code_info == 0
