"""Tuning parameters for the template analysis pipeline."""

from dataclasses import dataclass


@dataclass
class Config:
    """All knobs exposed by the pipeline, with the defaults used in the paper."""

    # Verse clustering variant to use (see v_clusterings table; 4 = "loose", threshold 0.75)
    clustering_id: int = 4

    # Minimum cluster frequency (number of verse occurrences) to include
    min_freq: int = 21

    # Cosine-similarity window for linking clusters into verse groups
    sim_threshold: float = 0.5   # lower bound (exclusive)
    sim_max: float = 0.99        # upper bound — excludes near-duplicates

    # Rows processed per batch during the chunked cosine-similarity pass
    chunk_size: int = 500

    # Maximum allowed verse-position gap between consecutive slots in an n-gram
    max_gap: int = 5

    # Longest n-gram template to consider
    max_ngram: int = 6

    # Minimum distinct poems required for an n-gram to become a template
    min_ngram_poems: int = 15
