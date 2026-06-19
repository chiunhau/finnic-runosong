"""
Verse group construction via TF-IDF cosine similarity.

A *verse group* is a set of verse clusters whose English translations are
semantically close enough to be treated as interchangeable variants.
Connected components of the pairwise similarity graph define the groups.
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from finnic_runosong.db import query
from finnic_runosong.build_templates.config import Config


def build_verse_groups(cfg: Config = Config()) -> list[list[int]]:
    """Return a list of verse groups, each group being a list of clust_ids.

    Steps:
      1. Load English translations for all clusters above *min_freq*.
      2. Build a TF-IDF document per cluster (concat of its translations).
      3. Compute pairwise cosine similarity in chunks.
      4. Link pairs in (sim_threshold, sim_max) and find connected components.
    """
    print('Loading verse translations …')
    df = query(f'''
        SELECT vc.clust_id, vt.verse_in_english
        FROM gizmosql.poetry.v_clust vc
        JOIN gizmosql.poetry.v_clust_freq vcf
            ON vc.clust_id = vcf.clust_id AND vc.clustering_id = vcf.clustering_id
        JOIN gizmosql.poetry.verse_poem vp ON vc.v_id = vp.v_id
        JOIN gizmosql.poetry.verses_translated vt
            ON vp.p_id = vt.p_id AND vp.pos = vt.pos
        WHERE vc.clustering_id = {cfg.clustering_id}
          AND vcf.freq >= {cfg.min_freq}
          AND vt.verse_in_english IS NOT NULL
          AND vc.clust_id != 310105
        ORDER BY vc.clust_id, vp.p_id, vp.pos
    ''')
    print(f'  {len(df):,} rows, {df["clust_id"].nunique():,} clusters')

    docs = (
        df.groupby('clust_id')['verse_in_english']
        .apply(lambda t: ' '.join(t.dropna()))
        .reset_index(name='doc')
    )
    clust_ids = docs['clust_id'].values

    print('Building TF-IDF matrix …')
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True)
    mat = tfidf.fit_transform(docs['doc'])
    print(f'  matrix: {mat.shape}')

    print('Computing chunked cosine similarity …')
    n = mat.shape[0]
    pairs = []
    for start in range(0, n, cfg.chunk_size):
        chunk = cosine_similarity(mat[start:start + cfg.chunk_size], mat)
        local_r, cols = np.where((chunk > cfg.sim_threshold) & (chunk < cfg.sim_max))
        global_r = local_r + start
        mask = cols > global_r  # upper triangle only — avoid duplicate pairs
        for lr, col, gr in zip(local_r[mask], cols[mask], global_r[mask]):
            pairs.append((int(clust_ids[gr]), int(clust_ids[col]), float(chunk[lr, col])))
        if start % 5000 == 0:
            print(f'  {start}/{n} rows, {len(pairs):,} pairs so far')

    pairs_df = pd.DataFrame(pairs, columns=['clust_id_1', 'clust_id_2', 'sim'])
    print(f'  {len(pairs_df):,} pairs above threshold')

    all_ids = pd.unique(pairs_df[['clust_id_1', 'clust_id_2']].values.ravel())
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    rows_idx = pairs_df['clust_id_1'].map(id_to_idx).values
    cols_idx = pairs_df['clust_id_2'].map(id_to_idx).values
    adj = csr_matrix(
        (pairs_df['sim'].values, (rows_idx, cols_idx)), shape=(len(all_ids),) * 2
    )
    _, labels = connected_components(adj + adj.T, directed=False)

    groups: dict[int, list] = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(all_ids[idx])
    group_list = sorted(groups.values(), key=len, reverse=True)
    print(f'  {len(group_list):,} verse groups covering {len(all_ids):,} clusters')
    return group_list


def build_group_members(group_list: list[list[int]], cfg: Config = Config()) -> pd.DataFrame:
    """Flatten group_list into a (clustering_id, group_id, clust_id) membership table."""
    df = pd.DataFrame([
        {'clust_id': cid, 'clustering_id': cfg.clustering_id, 'group_id': gid}
        for gid, clust_ids in enumerate(group_list)
        for cid in clust_ids
    ])
    return df
