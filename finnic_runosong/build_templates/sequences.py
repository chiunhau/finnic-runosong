"""
Verse labelling, n-gram occurrence extraction, and template aggregation.

Once every verse occurrence is tagged with a verse group ID (build_verse_group_labels),
a sliding window over each poem collects ordered n-tuples of groups
(build_poem_ngram_occurrences).  Those that recur across enough poems become
templates (build_templates).
"""

import pandas as pd

from finnic_runosong.db import query
from finnic_runosong.build_templates.config import Config


def gcols(n: int) -> list[str]:
    """Column names for an n-gram group sequence: ['group_id_1', ..., 'group_id_{n}']."""
    return [f'group_id_{k}' for k in range(1, n + 1)]


def pcols(n: int) -> list[str]:
    """Column names for an n-gram position sequence: ['pos_1', ..., 'pos_{n}']."""
    return [f'pos_{k}' for k in range(1, n + 1)]


def build_verse_group_labels(members: pd.DataFrame, cfg: Config = Config()) -> pd.DataFrame:
    """Tag every verse occurrence in the corpus with its verse group ID.

    Returns a DataFrame with columns:
      p_id, pos, clust_id, group_id
    """
    print('Loading verse-position sequence from DB …')
    id_list = ','.join(str(c) for c in members['clust_id'])
    df = query(f'''
        SELECT vp.p_id, vp.pos, vc.clust_id
        FROM gizmosql.poetry.verse_poem vp
        JOIN gizmosql.poetry.v_clust vc
            ON vp.v_id = vc.v_id AND vc.clustering_id = {cfg.clustering_id}
        WHERE vc.clust_id IN ({id_list})
        ORDER BY vp.p_id, vp.pos
    ''')
    df['group_id'] = df['clust_id'].map(members.set_index('clust_id')['group_id'])
    print(f'  {len(df):,} labelled verse positions across {df["p_id"].nunique():,} poems')
    return df[['p_id', 'pos', 'clust_id', 'group_id']]


def build_poem_ngram_occurrences(
    labels: pd.DataFrame, cfg: Config = Config()
) -> dict[int, pd.DataFrame]:
    """Find all ordered n-tuples of verse groups co-occurring within a poem.

    For each n in 2..max_ngram, a valid occurrence requires:
      - consecutive slots within the same poem
      - positional gap between adjacent slots ≤ max_gap
      - all group IDs in the n-tuple are distinct

    Returns a dict mapping n → DataFrame with columns:
      p_id, group_id_1[, ..., group_id_{n}], pos_1[, ..., pos_{n}]
    """
    print('Building n-gram occurrence tables …')
    base = (
        labels.sort_values(['p_id', 'pos'])[['p_id', 'pos', 'group_id']]
        .rename(columns={'group_id': 'group_id_1', 'pos': 'pos_1'})
        .reset_index(drop=True)
    )

    result = {}
    for n in range(2, cfg.max_ngram + 1):
        df = base.copy()
        for k in range(2, n + 1):
            df[f'group_id_{k}'] = df.groupby('p_id')['group_id_1'].shift(-(k - 1))
            df[f'pos_{k}'] = df.groupby('p_id')['pos_1'].shift(-(k - 1))

        gc = gcols(n)
        pc = pcols(n)
        df = df.dropna(subset=gc)
        df[gc + pc] = df[gc + pc].astype(int)

        for k in range(n - 1):
            df = df[df[pc[k + 1]] - df[pc[k]] <= cfg.max_gap]
        for i in range(n):
            for j in range(i + 1, n):
                df = df[df[gc[i]] != df[gc[j]]]

        result[n] = df[['p_id'] + gc + pc].reset_index(drop=True)
        print(f'  {n}-gram: {len(result[n]):,} occurrences')
    return result


def build_templates(
    ngrams: dict[int, pd.DataFrame], cfg: Config = Config()
) -> pd.DataFrame:
    """Aggregate n-gram occurrences into templates.

    A template is an ordered verse-group sequence that appears in at least
    min_ngram_poems distinct poems.  The result is a flat DataFrame with one
    row per template and columns:

      template_id, n_verses, n_poems,
      group_id_1, group_id_2, ..., group_id_{max_ngram}

    Unused slot columns are NULL for shorter templates.
    """
    print('Building templates …')
    rows = []
    for n in range(2, cfg.max_ngram + 1):
        gc = gcols(n)
        counts = (
            ngrams[n].groupby(gc)['p_id'].nunique()
            .reset_index(name='n_poems')
            .query('n_poems >= @cfg.min_ngram_poems')
        )
        print(f'  {n}-grams: {len(counts):,} templates')
        for _, row in counts.iterrows():
            entry = {'n_verses': n, 'n_poems': int(row['n_poems'])}
            for col in gc:
                entry[col] = int(row[col])
            for k in range(n + 1, cfg.max_ngram + 1):
                entry[f'group_id_{k}'] = None
            rows.append(entry)

    templates = pd.DataFrame(rows)
    templates.insert(0, 'template_id', range(len(templates)))
    print(f'  Total: {len(templates):,} templates')
    return templates
