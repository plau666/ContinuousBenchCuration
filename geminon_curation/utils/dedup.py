"""Re-export tools.dedup for geminon_curation."""
from tools.dedup import (  # noqa: F401
    text_hash,
    shingle,
    build_minhash,
    containment,
    exact_dedup,
    near_dedup,
)
