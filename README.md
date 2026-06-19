# Finnic Runosong Template Explorer

Tools for discovering and analysing recurring verse-group sequences ("templates") in the Finnish and Estonian oral poetry corpus, developed during the [Helsinki Digital Humanities Hackathon 2026](https://www.helsinki.fi/en/helsinki-centre-for-digital-humanities/digital-humanities-hackathon).

The corpus contains ~294 000 poems from four collections — `SKVR`, `erab`, `jr`, and `kr` — stored in a shared [Apache Arrow Flight SQL](https://arrow.apache.org/docs/format/FlightSql.html) database.

## Core concepts

### Verse groups

The database contains pre-computed **verse clusters** — sets of similar verse lines. This tool merges clusters into broader **verse groups** by computing TF-IDF cosine similarity on their English translations and finding connected components. A verse group collects clusters that express similar content and can be treated as interchangeable variants for sequence analysis.

### Templates

After tagging every verse occurrence with its verse group ID, a sliding window extracts all ordered n-tuples (n = 2–6) of verse groups within each poem. N-grams that recur in enough distinct poems become **templates** — recurring verse-group sequences in the corpus.

### Quality scores

Each template is scored on three dimensions:

| Score | Range | Meaning |
|-------|-------|---------|
| **Coverage** | 0–1 | Fraction of a template's poems *not* already explained by a longer parent template. Near 1 = independently meaningful; near 0 = always part of a longer sequence. |
| **Slot stability** | 0–1 | Per-slot, how fixed vs. interchangeable each position is. Near 1 = formulaic; near 0 = freely substitutable in context. |
| **Mobility** | 0–1 | Shannon entropy of where the template appears within poems (early / middle / late). Near 0 = anchored to a fixed position; near 1 = freely roaming. |

## Pipeline flow

```
DB (v_clust, verses_translated)
 │
 ├─ build_verse_groups          → cluster_group_members.parquet
 ├─ build_verse_group_labels    → verse_group_labels.parquet
 ├─ build_poem_ngram_occurrences → poem_{2..6}_gram_occurrences.parquet
 ├─ build_templates             ─┐
 ├─ build_template_edges         ├→ templates.parquet, template_edges.parquet
 ├─ build_template_coverage     ─┘
 ├─ build_template_slot_stability → template_slot_stability.parquet
 └─ build_template_mobility     → template_mobility.parquet
```

## Output files

All outputs are [Parquet](https://parquet.apache.org/) files.

| File | Description | Key columns |
|------|-------------|-------------|
| `cluster_group_members.parquet` | Maps each verse cluster to its verse group | `clust_id`, `group_id` |
| `verse_group_labels.parquet` | Every verse occurrence tagged with its group | `p_id`, `pos`, `clust_id`, `group_id` |
| `poem_{2..6}_gram_occurrences.parquet` | All ordered n-tuples of verse groups co-occurring within a poem (one file per n) | `p_id`, `group_id_1`..`group_id_n`, `pos_1`..`pos_n` |
| `templates.parquet` | Recurring sequences with poem counts and coverage | `template_id`, `n_verses`, `n_poems`, `coverage` |
| `template_edges.parquet` | Parent → child containment graph | `parent_template_id`, `child_template_id`, `offset` |
| `template_slot_stability.parquet` | Per-slot stability score | `template_id`, `position`, `stability` |
| `template_mobility.parquet` | Positional freedom score | `template_id`, `mobility`, `mobility_std` |

## Getting started

```bash
pip install -e .          # Python ≥ 3.10
cp .env.example .env      # fill in GizmoSQL credentials
export $(grep -v '^#' .env | xargs)
```

Run the full pipeline (writes to `data/` by default):

```bash
python -m finnic_runosong.build_templates
```

See all options with `python -m finnic_runosong.build_templates --help`.

## Package layout

```
finnic_runosong/
├── db.py                       # database connection and query helper
└── build_templates/
    ├── config.py               # Config dataclass (all tuning parameters)
    ├── groups.py               # verse group construction via TF-IDF
    ├── sequences.py            # verse labelling, n-gram extraction, template aggregation
    ├── analysis.py             # containment edges, coverage, stability, mobility
    └── __main__.py             # CLI entry point
```

## Raw data

Pre-built parquet tables and embeddings are available on Allas S3:
<https://a3s.fi/dhh26/index.html#poetry/>
