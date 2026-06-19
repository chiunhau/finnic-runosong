"""
Command-line entry point for the template-building pipeline.

Usage
-----
  python -m finnic_runosong.build_templates [--out-dir OUTPUT_DIR] [options]

All steps run sequentially and write parquet files to OUTPUT_DIR.
"""

import argparse
from pathlib import Path

import pandas as pd

from finnic_runosong.build_templates.config import Config
from finnic_runosong.build_templates.groups import build_verse_groups, build_group_members
from finnic_runosong.build_templates.sequences import build_verse_group_labels, build_poem_ngram_occurrences, build_templates
from finnic_runosong.build_templates.analysis import (
    build_template_edges,
    build_template_coverage,
    build_template_slot_stability,
    build_template_mobility,
)


def save(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)
    print(f'  saved {path}  ({len(df):,} rows)')


def main():
    parser = argparse.ArgumentParser(
        description='Build template analysis parquet tables for the Finnish/Estonian poetry corpus.',
    )
    parser.add_argument(
        '--out-dir', default='data', metavar='DIR',
        help='Directory to write output parquet files (default: data)',
    )
    parser.add_argument(
        '--clustering-id', type=int, default=Config.clustering_id,
        help=f'Verse clustering variant to use (default: {Config.clustering_id})',
    )
    parser.add_argument(
        '--min-freq', type=int, default=Config.min_freq,
        help=f'Minimum cluster frequency to include (default: {Config.min_freq})',
    )
    parser.add_argument(
        '--min-ngram-poems', type=int, default=Config.min_ngram_poems,
        help=f'Minimum poems per template (default: {Config.min_ngram_poems})',
    )
    args = parser.parse_args()

    cfg = Config(
        clustering_id=args.clustering_id,
        min_freq=args.min_freq,
        min_ngram_poems=args.min_ngram_poems,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Steps 1–2: group similar verse clusters via TF-IDF + connected components
    group_list = build_verse_groups(cfg)

    print('\n── cluster_group_members ──')
    members = build_group_members(group_list, cfg)
    save(members, out / 'cluster_group_members.parquet')

    # Step 3: tag every verse occurrence with its verse group ID
    print('\n── verse_group_labels ──')
    labels = build_verse_group_labels(members, cfg)
    save(labels, out / 'verse_group_labels.parquet')

    # Step 4: slide a window over poems to collect n-gram co-occurrences
    print('\n── poem n-gram occurrences ──')
    ngrams = build_poem_ngram_occurrences(labels, cfg)
    for n, df in ngrams.items():
        save(df, out / f'poem_{n}_gram_occurrences.parquet')

    # Steps 5–7: aggregate templates, build containment graph, compute coverage
    print('\n── templates ──')
    templates_df = build_templates(ngrams, cfg)

    print('\n── template_edges ──')
    edges, lookup = build_template_edges(templates_df, cfg)
    save(edges, out / 'template_edges.parquet')

    print('\n── coverage ──')
    coverage = build_template_coverage(templates_df, ngrams, edges, lookup, cfg)
    templates_df = templates_df.merge(coverage, on='template_id')
    save(templates_df, out / 'templates.parquet')

    # Step 8: measure how fixed each template slot is
    print('\n── slot stability ──')
    save(build_template_slot_stability(templates_df, cfg), out / 'template_slot_stability.parquet')

    # Step 9: measure how freely templates move within poems
    print('\n── template mobility ──')
    save(
        build_template_mobility(templates_df, ngrams, lookup, cfg),
        out / 'template_mobility.parquet',
    )

    print('\nDone.')


if __name__ == '__main__':
    main()
