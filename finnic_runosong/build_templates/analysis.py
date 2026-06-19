"""
Template containment graph and three quality scores: coverage, slot stability,
template mobility.

  build_template_edges    — containment graph: which templates are sub-sequences
                            of longer ones.

  build_template_coverage          — fraction of poems not already explained by a longer
                            parent template.
                            Near 1 = independently meaningful; near 0 = noise.

  build_template_slot_stability    — for each (template, slot), how fixed that position is
                            vs. sibling templates that differ only there.
                            Near 1 = formulaic; near 0 = interchangeable.

  build_template_mobility — entropy of positional distribution (early / middle /
                            late zones).
                            Near 0 = always anchored at same position.
                            Near 1 = freely roaming across poems.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from finnic_runosong.db import query
from finnic_runosong.build_templates.config import Config
from finnic_runosong.build_templates.sequences import gcols


def build_template_edges(
    templates: pd.DataFrame, cfg: Config = Config()
) -> tuple[pd.DataFrame, dict]:
    """Build the parent → child containment graph for templates.

    For every template T, find all contiguous sub-sequences of its group IDs
    that are themselves templates; those become children of T.

    Returns:
      edges_df  — DataFrame with columns: parent_template_id, child_template_id, offset
      lookup    — dict mapping group-tuple → template_id (reused in coverage step)
    """
    print('Building template edges …')
    gcol_names = [f'group_id_{k}' for k in range(1, cfg.max_ngram + 1)]

    def to_seq(row):
        return tuple(int(row[c]) for c in gcol_names[:int(row['n_verses'])])

    templates = templates.copy()
    templates['_seq'] = templates.apply(to_seq, axis=1)
    lookup = dict(zip(templates['_seq'], templates['template_id']))

    edges = []
    for _, row in templates.iterrows():
        seq = row['_seq']
        n = len(seq)
        for k in range(2, n):
            for offset in range(n - k + 1):
                child_id = lookup.get(seq[offset:offset + k])
                if child_id is not None:
                    edges.append({
                        'parent_template_id': int(row['template_id']),
                        'child_template_id':  int(child_id),
                        'offset':             offset,
                    })

    edges_df = pd.DataFrame(edges)
    n_parents = edges_df['parent_template_id'].nunique() if len(edges_df) else 0
    print(f'  {len(edges_df):,} edges across {n_parents:,} parent templates')
    return edges_df, lookup


def build_template_coverage(
    templates: pd.DataFrame,
    ngrams: dict[int, pd.DataFrame],
    edges: pd.DataFrame,
    lookup: dict,
    cfg: Config = Config(),
) -> pd.DataFrame:
    """Compute the fraction of each template's poems not covered by any longer parent.

    Returns a DataFrame with columns:
      template_id, n_poems_exclusive, coverage
    """
    print('Computing coverage …')

    poem_sets: dict[int, set] = {}
    for n in range(2, cfg.max_ngram + 1):
        gc = gcols(n)
        for key, grp in ngrams[n].groupby(gc):
            t = key if isinstance(key, tuple) else (key,)
            if (tid := lookup.get(t)) is not None:
                poem_sets[tid] = set(grp['p_id'])

    child_to_parents = (
        edges.groupby('child_template_id')['parent_template_id'].apply(list).to_dict()
    )

    rows = []
    for tid in templates['template_id']:
        my_poems = poem_sets.get(tid, set())
        parent_poems = set().union(
            *(poem_sets.get(p, set()) for p in child_to_parents.get(tid, []))
        )
        exclusive = my_poems - parent_poems
        rows.append({
            'template_id':       tid,
            'n_poems_exclusive': len(exclusive),
            'coverage':          len(exclusive) / len(my_poems) if my_poems else 0.0,
        })

    result = pd.DataFrame(rows)
    print(f'  coverage  mean={result["coverage"].mean():.2f}  '
          f'median={result["coverage"].median():.2f}')
    return result


def build_template_slot_stability(
    templates: pd.DataFrame, cfg: Config = Config()
) -> pd.DataFrame:
    """Measure how fixed each slot is within its template.

    For each (template T, position i): find all *sibling* templates that share
    T's group IDs at every position except i.  Stability is T's poem count
    divided by the total across T and all siblings.

    Returns a DataFrame with columns:
      template_id, position, stability, n_siblings, n_poems_context
    """
    print('Computing stability …')
    rows = []
    for n in range(2, cfg.max_ngram + 1):
        subset = templates[templates['n_verses'] == n]
        if subset.empty:
            continue
        gc = [f'group_id_{k}' for k in range(1, n + 1)]

        for pos_i in range(n):
            context_cols = [c for j, c in enumerate(gc) if j != pos_i]
            context_totals = (
                subset.groupby(context_cols)['n_poems']
                .sum().reset_index(name='n_poems_context')
            )
            sibling_counts = (
                subset.groupby(context_cols)['template_id']
                .count().reset_index(name='n_siblings')
            )
            merged = (
                subset
                .merge(context_totals, on=context_cols)
                .merge(sibling_counts, on=context_cols)
                .assign(
                    stability=lambda d: d['n_poems'] / d['n_poems_context'],
                    position=pos_i,
                )
            )
            rows.append(
                merged[['template_id', 'position', 'stability', 'n_siblings', 'n_poems_context']]
            )

    result = pd.concat(rows, ignore_index=True)
    print(f'  {len(result):,} (template, position) rows  '
          f'stability mean={result["stability"].mean():.2f}')
    return result


def _resolve_template_ids(
    ngrams: dict[int, pd.DataFrame], lookup: dict, cfg: Config,
) -> pd.DataFrame:
    """Tag each n-gram occurrence row with its template_id (if one exists)."""
    parts = []
    for n in range(2, cfg.max_ngram + 1):
        df = ngrams[n].copy()
        gc = gcols(n)
        df['_key'] = list(zip(*[df[c] for c in gc]))
        df['_tid'] = df['_key'].apply(lookup.get)
        df = df.dropna(subset=['_tid'])
        df['_tid'] = df['_tid'].astype(int)
        parts.append(df[['p_id', 'pos_1', '_tid']])
    return pd.concat(parts, ignore_index=True)


def build_template_mobility(
    templates: pd.DataFrame,
    ngrams: dict[int, pd.DataFrame],
    lookup: dict,
    cfg: Config = Config(),
) -> pd.DataFrame:
    """Measure how freely each template moves within poems.

    For every occurrence, compute the normalised start position
    (verse_pos / poem_length) and bin into early/middle/late terciles.
    Mobility is the Shannon entropy of that distribution, normalised
    to [0, 1].

    Returns a DataFrame with columns:
      template_id, mobility, mobility_std, n_occurrences
    """
    print('Computing mobility …')

    print('  loading poem_stats …')
    poem_stats = query('SELECT p_id, nverses FROM gizmosql.poetry.poem_stats')
    poem_len = poem_stats.set_index('p_id')['nverses'].to_dict()

    tagged = _resolve_template_ids(ngrams, lookup, cfg)
    tagged['_nverses'] = tagged['p_id'].map(poem_len)
    tagged = tagged[tagged['_nverses'].notna() & (tagged['_nverses'] > 0)]
    tagged['_norm_pos'] = tagged['pos_1'] / tagged['_nverses']

    template_positions: dict[int, list[float]] = defaultdict(list)
    for tid, grp in tagged.groupby('_tid'):
        template_positions[int(tid)].extend(grp['_norm_pos'].tolist())

    MAX_ENT = np.log2(3)

    rows = []
    for tid in templates['template_id']:
        positions = template_positions.get(tid, [])
        n_occ = len(positions)

        if n_occ < 2:
            rows.append({
                'template_id': tid,
                'mobility': np.nan,
                'mobility_std': np.nan,
                'n_occurrences': n_occ,
            })
            continue

        arr = np.array(positions)
        counts = np.array([
            np.sum(arr < 1 / 3),
            np.sum((arr >= 1 / 3) & (arr < 2 / 3)),
            np.sum(arr >= 2 / 3),
        ], dtype=float)
        probs = counts / counts.sum()
        probs_nz = probs[probs > 0]
        entropy = -np.sum(probs_nz * np.log2(probs_nz))

        rows.append({
            'template_id': tid,
            'mobility': float(entropy / MAX_ENT),
            'mobility_std': float(np.std(arr)),
            'n_occurrences': n_occ,
        })

    result = pd.DataFrame(rows)
    finite = result['mobility'].dropna()
    print(f'  {len(result):,} templates  '
          f'mobility mean={finite.mean():.2f}  '
          f'median={finite.median():.2f}')
    return result
