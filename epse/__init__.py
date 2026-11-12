"""EPSE: Efficient Protein Search Engine for Resource-Constrained Systems.

Uses Protein Language Models (ESM-2) and modern e-means clustering to enable fast,
low-memory similarity search over large protein databases on CPU.
"""

__version__ = "0.2.0"

from .fasta import parse_fasta, UniversalFastaEntry
from .embedder import ProteinEmbedder
from .emeans import EMeansClustering
from .storage import IndexStorage
from .search import ProteinSearchEngine, SearchResult, SearchMetrics

__all__ = [
    "parse_fasta",
    "UniversalFastaEntry",
    "ProteinEmbedder",
    "EMeansClustering",
    "IndexStorage",
    "ProteinSearchEngine",
    "SearchResult",
    "SearchMetrics",
]
