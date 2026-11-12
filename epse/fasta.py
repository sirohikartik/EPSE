"""Universal FASTA parser supporting arbitrary protein databases.

Supports UniProtKB (Swiss-Prot/TrEMBL), NCBI (RefSeq/GenBank), Ensembl, PDB,
PIR, and generic or user-defined FASTA headers. Also supports transparent
reading from gzipped files (.gz).
"""

from __future__ import annotations

import gzip
import io
import os
import re
from dataclasses import dataclass
from typing import Generator, Union, TextIO


@dataclass
class UniversalFastaEntry:
    """Represents a standardized protein entry extracted from any FASTA record."""
    accession: str
    entry_name: str
    description: str
    organism: str
    seq: str

    def to_dict(self) -> dict[str, Union[str, int]]:
        return {
            "accession": self.accession,
            "entry_name": self.entry_name,
            "description": self.description,
            "organism": self.organism,
            "seq": self.seq,
            "seq_len": len(self.seq),
        }

    def __getitem__(self, item: str):
        if item == "seq_len":
            return len(self.seq)
        return getattr(self, item)

    def get(self, item: str, default=None):
        try:
            return self[item]
        except AttributeError:
            return default


def _clean_sequence(seq_fragments: list[str]) -> str:
    """Combines sequence fragments, converts to uppercase and removes invalid characters."""
    raw = "".join(seq_fragments).upper()
    # Strip whitespace, newlines, and trailing stop codon symbols
    cleaned = re.sub(r"\s+", "", raw).rstrip("*")
    return cleaned


def _parse_header(header_line: str) -> tuple[str, str, str, str]:
    """Parses diverse header formats into (accession, entry_name, description, organism)."""
    header = header_line.lstrip(">").strip()
    if not header:
        return ("unknown", "unknown", "", "unknown")

    accession = "unknown"
    entry_name = "unknown"
    description = ""
    organism = "unknown"

    # 1. Pipe-separated headers (e.g. UniProt, NCBI gi/ref, PIR)
    if "|" in header:
        parts = header.split("|")
        prefix = parts[0].lower()

        # Standard 3-part pipe style: >db|accession|entry_name description OS=...
        # Also covers UniProt (>sp|..., >tr|...) and generic pipe formats with 3+ parts
        if prefix in ("sp", "tr", "db") or (len(parts) >= 3 and prefix not in ("pdb", "gi", "ref")):
            accession = parts[1] if len(parts) > 1 else "unknown"
            rest = parts[2] if len(parts) > 2 else ""
            if " " in rest:
                entry_name, desc_part = rest.split(" ", 1)
            else:
                entry_name, desc_part = rest, ""

            # Check for organism via OS= or [Organism]
            if " OS=" in desc_part:
                desc_pre, os_post = desc_part.split(" OS=", 1)
                description = desc_pre.strip()
                match_org = re.split(r"\s+(?:OX|GN|PE|SV)=", os_post, maxsplit=1)
                organism = match_org[0].strip()
            elif "[" in desc_part and "]" in desc_part:
                org_match = re.search(r"\[(.*?)\]", desc_part)
                if org_match:
                    organism = org_match.group(1).strip()
                    description = re.sub(r"\[.*?\]", "", desc_part).strip()
                else:
                    description = desc_part.strip()
            else:
                description = desc_part.strip()

            return accession, entry_name, description, organism

        # NCBI style: >gi|12345|ref|NP_001234.1| description [Organism]
        if "ref" in parts or "gi" in parts:
            if "ref" in parts:
                ref_idx = parts.index("ref")
                if len(parts) > ref_idx + 1:
                    accession = parts[ref_idx + 1]
            elif len(parts) > 1:
                accession = parts[1]

            rest = parts[-1] if len(parts) > 1 else ""
            entry_name = accession
            description = rest.strip()
            # Extract organism from bracket [Organism]
            org_match = re.search(r"\[(.*?)\]", description)
            if org_match:
                organism = org_match.group(1).strip()
                description = re.sub(r"\[.*?\]", "", description).strip()
            return accession, entry_name, description, organism

        # PDB style or PIR style: >pdb|1ABC|A or >pir||A00001
        if prefix == "pdb":
            pdb_id = parts[1] if len(parts) > 1 else "unknown"
            chain_and_desc = parts[2] if len(parts) > 2 else ""
            if " " in chain_and_desc:
                chain, desc_remainder = chain_and_desc.split(" ", 1)
            else:
                chain, desc_remainder = chain_and_desc, ""
            accession = f"{pdb_id}_{chain}".strip("_")
            entry_name = accession
            extra = (" " + parts[3].strip()) if len(parts) > 3 else ""
            full_desc = (desc_remainder + extra).strip()

            if " OS=" in full_desc:
                desc_pre, os_post = full_desc.split(" OS=", 1)
                description = desc_pre.strip()
                organism = re.split(r"\s+(?:OX|GN|PE|SV)=", os_post, maxsplit=1)[0].strip()
            else:
                description = full_desc
            return accession, entry_name, description, organism

        # Generic pipe format
        accession = parts[1] if len(parts) > 1 else parts[0]
        entry_name = accession
        description = " ".join(parts[2:]).strip() if len(parts) > 2 else ""
        return accession, entry_name, description, organism

    # 2. Non-pipe formats: Ensembl, NCBI simple, PDB simple, or generic space-separated
    first_token, *rest_tokens = header.split(None, 1)
    accession = first_token.strip()
    entry_name = accession
    remainder = rest_tokens[0].strip() if rest_tokens else ""

    # Check for organism in brackets e.g. [Homo sapiens]
    org_match = re.search(r"\[(.*?)\]", remainder)
    if org_match:
        organism = org_match.group(1).strip()
        description = re.sub(r"\[.*?\]", "", remainder).strip()
    elif " OS=" in remainder:
        desc_pre, os_post = remainder.split(" OS=", 1)
        description = desc_pre.strip()
        match_org = re.split(r"\s+(?:OX|GN|PE|SV)=", os_post, maxsplit=1)
        organism = match_org[0].strip()
    else:
        description = remainder

    # Ensembl format tags: gene_symbol:..., description:...
    if "gene_symbol:" in description or "description:" in description:
        desc_match = re.search(r"description:(.*?)(?:\[|$)", remainder)
        if desc_match:
            description = desc_match.group(1).strip()

    return accession, entry_name, description, organism


def parse_fasta(
    source: Union[str, TextIO, io.StringIO]
) -> Generator[UniversalFastaEntry, None, None]:
    """Streams protein entries from a FASTA file or stream.

    Supports arbitrary protein databases, multiline sequences, and .gz compression.

    Args:
        source: Path to FASTA file (str) or an open text file stream.

    Yields:
        UniversalFastaEntry instances containing parsed metadata and sequences.
    """
    if isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"FASTA file not found: {source}")

        if source.endswith(".gz"):
            file_obj = gzip.open(source, "rt", encoding="utf-8", errors="replace")
        else:
            file_obj = open(source, "r", encoding="utf-8", errors="replace")
        close_needed = True
    else:
        file_obj = source
        close_needed = False

    try:
        current_header = None
        seq_fragments: list[str] = []

        for line in file_obj:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_header is not None:
                    cleaned_seq = _clean_sequence(seq_fragments)
                    if cleaned_seq:
                        acc, entry, desc, org = _parse_header(current_header)
                        yield UniversalFastaEntry(
                            accession=acc,
                            entry_name=entry,
                            description=desc,
                            organism=org,
                            seq=cleaned_seq,
                        )
                current_header = line
                seq_fragments = []
            else:
                seq_fragments.append(line)

        if current_header is not None:
            cleaned_seq = _clean_sequence(seq_fragments)
            if cleaned_seq:
                acc, entry, desc, org = _parse_header(current_header)
                yield UniversalFastaEntry(
                    accession=acc,
                    entry_name=entry,
                    description=desc,
                    organism=org,
                    seq=cleaned_seq,
                )
    finally:
        if close_needed:
            file_obj.close()
