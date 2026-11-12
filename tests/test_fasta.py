"""Tests for the universal FASTA parser supporting arbitrary protein databases."""

import io
import pytest
from epse.fasta import parse_fasta, _parse_header, UniversalFastaEntry


def test_parse_uniprot_sprot_header():
    header = ">sp|P12345|A4_HUMAN Amyloid-beta A4 protein OS=Homo sapiens OX=9606 GN=APP PE=1 SV=3"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "P12345"
    assert entry == "A4_HUMAN"
    assert desc == "Amyloid-beta A4 protein"
    assert org == "Homo sapiens"


def test_parse_uniprot_trembl_header():
    header = ">tr|A0A024RBG1|A0A024RBG1_HUMAN NUDT4B protein OS=Homo sapiens OX=9606 GN=NUDT4B PE=3 SV=1"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "A0A024RBG1"
    assert entry == "A0A024RBG1_HUMAN"
    assert desc == "NUDT4B protein"
    assert org == "Homo sapiens"


def test_parse_ncbi_refseq_pipe_header():
    header = ">gi|4501865|ref|NP_000676.1| alcohol dehydrogenase 1A [Homo sapiens]"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "NP_000676.1"
    assert org == "Homo sapiens"
    assert "alcohol dehydrogenase" in desc


def test_parse_ncbi_simple_header():
    header = ">NP_001300940.1 cytochrome P450 2D6 [Mus musculus]"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "NP_001300940.1"
    assert org == "Mus musculus"
    assert "cytochrome P450" in desc


def test_parse_ensembl_header():
    header = ">ENSP00000369497 pep chromosome:GRCh38:1:1:100:1 gene:ENSG00000198888 description:NADH dehydrogenase [Homo sapiens]"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "ENSP00000369497"
    assert org == "Homo sapiens"
    assert "NADH dehydrogenase" in desc


def test_parse_pdb_header():
    header = ">pdb|4HHB|A Hemoglobin subunit alpha OS=Homo sapiens"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "4HHB_A"
    assert org == "Homo sapiens"
    assert "Hemoglobin" in desc


def test_parse_generic_custom_header():
    header = ">synthetic_construct_01 Designed helical peptide"
    acc, entry, desc, org = _parse_header(header)
    assert acc == "synthetic_construct_01"
    assert entry == "synthetic_construct_01"
    assert desc == "Designed helical peptide"
    assert org == "unknown"


def test_stream_multiline_and_whitespace():
    fasta_data = """
>sp|P01308|INS_HUMAN Insulin OS=Homo sapiens OX=9606 GN=INS PE=1 SV=1
MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKT
RREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENY
CN*

>custom_prot_2 Test protein
M K T V R
Q E R L K
"""
    stream = io.StringIO(fasta_data)
    entries = list(parse_fasta(stream))

    assert len(entries) == 2

    # Check first entry
    assert entries[0].accession == "P01308"
    assert entries[0].entry_name == "INS_HUMAN"
    assert entries[0].organism == "Homo sapiens"
    # Ensure whitespace removed and asterisk stripped
    assert "*" not in entries[0].seq
    assert "\n" not in entries[0].seq
    assert entries[0].seq.startswith("MALWMR")
    assert entries[0].seq.endswith("CN")

    # Check second entry
    assert entries[1].accession == "custom_prot_2"
    assert entries[1].seq == "MKTVRQERLK"
    assert entries[1]["seq_len"] == 10
    assert entries[1].to_dict()["accession"] == "custom_prot_2"
