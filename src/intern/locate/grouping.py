"""Terms, TF-IDF vectors and keywords for the files: what the tree's affinity and every card's keywords come from.
Pure: texts in, vectors and words out, no model calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import csr_matrix


#: Places files for grouping: texts by path in, a unit vector (numpy) by path out.
Vectorizer = Callable[[Mapping[str, str]], Mapping[str, "np.ndarray"]]

_WORD = re.compile(r"[a-z][a-z0-9_]{2,}")


def tokens(text: str) -> list[str]:
    """The terms of a text: lowercase words of three or more characters, without stop words and placeholders. CJK
    text contributes none: character bigrams were tried and cost 0.06 of hit@1 on the office localization set
    (the bigrams are not words, and they crowd the English keywords off a card)."""
    return [w.strip("_") for w in _WORD.findall(text.lower()) if w not in _STOP and not _JUNK.match(w)]


#: Placeholder column names, file types and the like: present everywhere, telling nothing.
_JUNK = re.compile(
    r"^(col\d+|column\d+|empty|jpe?g|png|gif|bmp|webp|pdf|xlsx?|xlsm|docx?|pptx?|csv|tsv|txt|html?|json|md|yaml|yml)$"
)
#: Words the index's own rendering puts in every table or page, which would otherwise be every folder's keywords.
_BOILERPLATE = [
    "rows",
    "row",
    "columns",
    "column",
    "table",
    "tab",
    "sheet",
    "sheet1",
    "page",
    "pages",
    "col",
    "lines",
    "image",
]
_STOP = frozenset(
    [
        *_BOILERPLATE,
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "not",
        "are",
        "was",
        "were",
        "will",
        "have",
        "has",
        "been",
        "its",
        "into",
        "than",
        "then",
        "them",
        "they",
        "there",
        "their",
        "which",
        "while",
        "where",
        "when",
        "what",
        "who",
        "whom",
        "whose",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "any",
        "because",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "further",
        "here",
        "how",
        "into",
        "just",
        "more",
        "most",
        "much",
        "must",
        "now",
        "off",
        "once",
        "only",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "some",
        "such",
        "too",
        "under",
        "until",
        "very",
        "you",
        "your",
        "import",
        "def",
        "class",
        "return",
        "self",
        "none",
        "true",
        "false",
    ]
)


def tfidf(texts: Mapping[str, str]) -> tuple[list[str], csr_matrix, list[str]]:
    """Unit TF-IDF rows for ``texts``: (paths, sparse matrix, vocabulary). Terms are lowercase words of three or more
    characters that are not stop words and appear in at least two texts (a word one file has links it to nothing).
    Sparse, as a file has a few hundred of the vocabulary's tens of thousands of terms: dense, 16k files of
    audit-app took 12 GB."""
    import numpy as np
    from scipy.sparse import csr_matrix, diags

    paths = sorted(texts)
    bags = [Counter(tokens(texts[p])) for p in paths]
    df = Counter(w for bag in bags for w in bag)
    vocab = sorted(w for w, n in df.items() if n >= 2) if len(paths) > 1 else sorted(df)
    column = {w: i for i, w in enumerate(vocab)}
    n = len(paths)
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for row, bag in enumerate(bags):
        for w, tf in bag.items():
            if w in column:
                rows.append(row)
                cols.append(column[w])
                weights.append((1 + math.log(tf)) * math.log((1 + n) / (1 + df[w])) + 1)
    matrix = csr_matrix((weights, (rows, cols)), shape=(n, len(vocab)), dtype=np.float64)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return paths, csr_matrix(diags(np.divide(1.0, norms, out=np.zeros(n), where=norms > 0)) @ matrix), vocab


def top_terms(row: csr_matrix, vocab: list[str], n: int) -> list[str]:
    """The ``n`` heaviest terms of a one-row matrix, ties in vocabulary order."""
    import numpy as np

    order = np.lexsort((row.indices, -row.data))[:n]
    return [vocab[row.indices[i]] for i in order if row.data[i] > 0]
