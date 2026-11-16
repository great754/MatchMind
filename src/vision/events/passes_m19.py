"""Versioned receiver-at-release evidence without changing the v1 thresholds."""
from .passes import PassConfig,detect_passes


def existing_m9_receiver(first,receiver):
    matches=[p for p in first['people'].values() if p['entity']==receiver['entity']]
    return matches[0] if len(matches)==1 else None


def detect_m19(recording,worlds,balls,fps,cfg=None):
    # Resolve only an already shared M9 identity. Different/ambiguous identities
    # are never merged; endpoint segment references and all gates stay explicit.
    return detect_passes(recording,worlds,balls,fps,cfg or PassConfig(),receiver_lookup=existing_m9_receiver)
