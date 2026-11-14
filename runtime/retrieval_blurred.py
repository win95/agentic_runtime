"""Code-blurred fake embeddings — Phase 1 measurability setup.

The Phase 0 BoW fake literally contains the distinguishing codes
(``F-204`` vs ``F-206``, ``O21`` vs ``O22``), so it saturates
(Recall@6 1.000, sibling nDCG@6 1.000) and the graph arm cannot beat
it. Real embedding models blur codes: near-identical maintenance
notes about sibling assets get near-identical vectors.

``CodeBlurredFakeEmbeddings`` simulates that failure mode on top of
the Phase 0 BoW vectors: every token containing a digit is dropped
before vectorising — asset codes (``O21``), fault codes (``F-204``),
component codes (``motor-M2``), temperatures (``92C``) and
measurements (``3mm``, ``4.5``) all vanish, leaving only topic words.
Two sibling faults that differ only in code + temperature then have
identical vectors, so the vector rank cannot tell them apart and
only the graph arm (or an exact-code feature) can.

This is a *simulation* for judging the Phase 1 exit criterion, not
a claim about any real embedding model. It reuses the Phase 0
vocabulary and ``FakeQdrantClient`` unchanged.
"""

from __future__ import annotations

import math

from runtime.retrieval_baseline import DeterministicFakeEmbeddings, tokenize


class CodeBlurredFakeEmbeddings(DeterministicFakeEmbeddings):
    """BoW vectors with every digit-bearing token dropped."""

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * len(self.vocabulary)
        for token in tokenize(text):
            if any(ch.isdigit() for ch in token):
                continue
            idx = self._index.get(token)
            if idx is not None:
                vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            return vec
        return [v / norm for v in vec]
