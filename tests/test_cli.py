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

    # Run search command
    args_search = parser.parse_args([
        "search",
        "MALWMRLLPLLALLALWGPDPAAA",
        "--data-dir", data_dir,
        "--top-k", "1",
        "--use-mock",
        "--format", "json",
    ])
    code_search = args_search.func(args_search)
    assert code_search == 0

    # Run info command
    args_info = parser.parse_args([
        "info",
        "--data-dir", data_dir,
    ])
    code_info = args_info.func(args_info)
    assert code_info == 0
