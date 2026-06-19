"""Template analysis pipeline for the Finnish/Estonian poetry corpus."""

from finnic_runosong.build_templates.config import Config
from finnic_runosong.build_templates.groups import build_verse_groups, build_group_members
from finnic_runosong.build_templates.sequences import build_verse_group_labels, build_poem_ngram_occurrences, build_templates
from finnic_runosong.build_templates.analysis import (
    build_template_edges,
    build_template_coverage,
    build_template_slot_stability,
    build_template_mobility,
)

__all__ = [
    'Config',
    'build_verse_groups',
    'build_group_members',
    'build_verse_group_labels',
    'build_poem_ngram_occurrences',
    'build_templates',
    'build_template_edges',
    'build_template_coverage',
    'build_template_slot_stability',
    'build_template_mobility',
]
