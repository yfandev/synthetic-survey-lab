"""The search tree over the files: folders first, balanced so every node is one card System 1 can judge.

A subtree of at most LEAF_FILES files is one leaf group. Loose files beyond that are cut into communities of the
affinity graph; a unit smaller than LEAF_MIN merges with its most tied sibling, so there are no singletons; a node
with more than FAN_MAX children gets intermediate groups by agglomerating the most tied siblings. Affinity is what
ties two files: for Python, a direct import either way and the files that share its rarer import targets; for every
file, the nearest neighbours of its TF-IDF vector. No embeddings, no model.

Ported from the NDI locate prototype, with the import affinity made one source among two so the
same tree serves a repository and an office workspace.
"""

from __future__ import annotations

import ast
import itertools
import math
import os
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from intern.locate import cards, grouping

if TYPE_CHECKING:
    import numpy as np
    from scipy.sparse import csr_matrix

#: Files per leaf group at most, and the size below which a unit merges into a sibling.
LEAF_FILES = 20
LEAF_MIN = 4
#: Children per inner node at most.
FAN_MAX = 12
#: TF-IDF neighbours that tie a file, and the import-target neighbours that tie a Python file.
NEIGHBOURS = 8
IMPORT_NEIGHBOURS = 8
#: Similarities in memory at once while finding the neighbours (blocks of files against all, one block a thread):
#: 128 MB of floats. The blocks run on up to SIMILARITY_THREADS threads (scipy's product releases the GIL): 16k
#: files of audit-app in 0.7 s instead of 4.8 s.
SIMILARITY_CELLS = 1 << 24
SIMILARITY_THREADS = 8
#: A module imported by more than this share of files ties nothing (too common to mean affinity).
COMMON_TARGET_SHARE = 0.05
RESOLUTION_STEP = 1.5
MAX_RESOLUTION = 64.0
#: Central files an inner card names per subgroup, and for the whole subtree.
SUBGROUP_FILES = 3
SUBTREE_FILES = 6
TOP_NAMES = 5
CARD_CHARS = 4000
LABEL_NAMES = 4
#: The files whose summary states what their folder is for, in the order they are asked. Not AGENTS.md or CLAUDE.md:
#: they tell an agent how to work there ("Follow the root guidance"), not what the folder is.
DESCRIBING = ("README.md", "README.rst", "README", "pyproject.toml", "package.json", "__init__.py")
#: Characters of the root card's purpose, and of each described folder a purpose lists.
PURPOSE_CHARS = 600
ENTRY_CHARS = 140
#: A docstring left as a placeholder ("tmp comment: TODO remove this") says nothing about its folder.
_PLACEHOLDER = re.compile(r"\b(TODO|FIXME|XXX)\b|^tmp\b", re.IGNORECASE)
#: Loose files of a folder are a series when this share of them are named by one template with numbers in it
#: (treasury_bulletin_1941_01.pdf, invoice-0042.pdf): they are cut in the order of those numbers, not by content.
SERIES_SHARE = 0.8
_DIGITS = re.compile(r"\d+")

Affinity = dict[str, dict[str, float]]


class Node(BaseModel):
    """A node of the tree: a leaf group of files, or an inner node over child nodes. ``key`` is unique; ``label`` is
    what the card and the agent see."""

    key: str
    label: str
    kind: Literal["leaf", "inner"]
    files: list[str]
    children: list[str] = []
    card: str = ""


# -- affinity ------------------------------------------------------------------------------------------------------


def import_names(root: Path, path: str) -> list[str]:
    """The dotted names a Python file imports, relative imports made absolute from its path, in the order found.
    The index keeps them with the file's entry, so a refresh parses only the files that changed."""
    try:
        tree = ast.parse((root / path).read_text(errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return []
    package = path[:-3].split("/")[:-1]
    names: list[str] = []
    for node in _statements(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                up = package[: len(package) - (node.level - 1)] if node.level > 1 else package
                base = ".".join([*up, base] if base else up)
            names += [f"{base}.{a.name}" for a in node.names]
    return names


def _statements(tree: ast.Module) -> Iterator[ast.AST]:
    """The module's statements, nested ones included, in `ast.walk`'s order. An import is a statement, so this finds
    what `ast.walk` finds without visiting the expressions, most of a module's nodes."""
    todo: deque[ast.AST] = deque([tree])
    while todo:
        node = todo.popleft()
        yield node
        for field in ("body", "handlers", "orelse", "finalbody", "cases"):  # the order of every node's _fields
            block = getattr(node, field, None)
            if isinstance(block, list):
                todo.extend(block)


def import_targets(paths: list[str], names: Mapping[str, list[str]]) -> dict[str, set[str]]:
    """For each Python file, the repository files it imports (``names``, from `import_names`, resolved to the longest
    matching module; a top-level package's __init__ is too generic to count)."""
    py = [p for p in paths if p.endswith(".py")]
    mod: dict[str, str] = {}
    for p in py:
        m = p[:-3].replace("/", ".").removesuffix(".__init__")
        mod[m] = p
        for prefix in ("lib.", "src."):
            if m.startswith(prefix):
                mod.setdefault(m[len(prefix) :], p)
    generic = {p for m, p in mod.items() if "." not in m}

    def resolve(name: str) -> str | None:
        parts = name.split(".")
        for k in range(len(parts), 0, -1):
            q = mod.get(".".join(parts[:k]))
            if q:
                return None if q in generic else q
        return None

    out: dict[str, set[str]] = {}
    for p in py:
        targets: set[str] = set()
        for name in names.get(p, []):
            q = resolve(name)
            if q and q != p:
                targets.add(q)
        out[p] = targets
    return out


def affinity(
    paths: list[str], names: Mapping[str, list[str]], vectors: Mapping[str, np.ndarray | csr_matrix]
) -> tuple[Affinity, dict[str, float]]:
    """Undirected ties between files, and each file's centrality. Python files (``names``: what each imports): a
    direct import (weight 1) plus the IMPORT_NEIGHBOURS most similar files by shared rare import targets (Jaccard).
    Every file: its NEIGHBOURS nearest vectors (cosine; a vector is a one-row sparse matrix or a numpy array),
    found a block of files at a time so memory stays within SIMILARITY_CELLS however many files there are.
    Centrality is how many files import a file, else the sum of its ties."""
    import numpy as np
    from scipy import sparse

    adj: Affinity = defaultdict(lambda: defaultdict(float))
    imports = import_targets(paths, names)
    indeg: Counter[str] = Counter()
    holders: dict[str, list[str]] = defaultdict(list)
    for p, targets in imports.items():
        for q in targets:
            indeg[q] += 1
            holders[q].append(p)
            adj[p][q] = max(adj[p][q], 1.0)
            adj[q][p] = max(adj[q][p], 1.0)
    cap = max(3, int(len(paths) * COMMON_TARGET_SHARE))
    rare = {p: {q for q in ts if len(holders[q]) <= cap} for p, ts in imports.items()}
    for p, ts in rare.items():
        if not ts:
            continue
        candidates: Counter[str] = Counter()
        for q in ts:
            for other in holders[q]:
                if other != p:
                    candidates[other] += 1
        ranked = sorted(((c / len(ts | rare.get(o, set())), o) for o, c in candidates.items()), reverse=True)
        for s, o in ranked[:IMPORT_NEIGHBOURS]:
            adj[p][o] += s
            adj[o][p] += s
    present = [p for p in paths if p in vectors]
    if len(present) > 1:
        rows = [vectors[p] for p in present]
        x = sparse.csr_matrix(sparse.vstack(rows) if sparse.issparse(rows[0]) else np.asarray(rows, dtype=np.float64))
        xt = sparse.csr_matrix(x.T)
        k = min(NEIGHBOURS, len(present) - 1)
        threads = min(SIMILARITY_THREADS, os.cpu_count() or 1)
        block = max(1, SIMILARITY_CELLS // (len(present) * threads))

        def nearest(start: int) -> tuple[np.ndarray, np.ndarray]:
            sims = sparse.csr_matrix(x[start : start + block] @ xt).toarray()
            own = np.arange(len(sims))
            sims[own, start + own] = -1.0
            near = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            return near, np.take_along_axis(sims, near, axis=1)

        with ThreadPoolExecutor(threads) as pool:
            for start, (near, sims) in zip(
                range(0, len(present), block), pool.map(nearest, range(0, len(present), block)), strict=True
            ):
                for r in range(len(near)):
                    p = present[start + r]
                    for j, s in zip(near[r], sims[r], strict=True):
                        if s > 0:
                            adj[p][present[j]] += float(s)
                            adj[present[j]][p] += float(s)
    centrality = {p: float(indeg[p]) if indeg[p] else sum(adj[p].values()) for p in paths}
    return adj, centrality


def _aff(adj: Affinity, a: list[str], b: set[str]) -> float:
    return sum(w for p in a for q, w in adj.get(p, {}).items() if q in b)


# -- communities ----------------------------------------------------------------------------------------------------


def communities(paths: list[str], adj: Affinity) -> list[list[str]]:
    """Louvain communities of the affinity graph over ``paths``, a community too large cut again at a finer resolution
    then halved in path order; a lone file joins the community it is most tied to while that has room; files tied to
    nothing are packed in path order."""
    import networkx as nx

    if len(paths) <= LEAF_FILES:
        return [paths]
    index = {p: i for i, p in enumerate(paths)}
    g = nx.Graph()
    g.add_nodes_from(range(len(paths)))
    for p in paths:
        for q, w in adj.get(p, {}).items():
            if q in index and index[q] > index[p]:
                g.add_edge(index[p], index[q], weight=w)
    parts: list[list[int]] = []
    todo = [(list(range(len(paths))), 1.0)]
    while todo:
        nodes, resolution = todo.pop()
        if len(nodes) <= LEAF_FILES:
            parts.append(nodes)
            continue
        sub = g.subgraph(nodes)
        found = (
            [sorted(c) for c in nx.community.louvain_communities(sub, weight="weight", resolution=resolution, seed=0)]
            if sub.number_of_edges()
            else [nodes]
        )
        if len(found) == 1:
            if resolution < MAX_RESOLUTION and sub.number_of_edges():
                todo.append((nodes, resolution * 2))
                continue
            half = len(nodes) // 2
            found = [nodes[:half], nodes[half:]]
        todo.extend((c, resolution * RESOLUTION_STEP) for c in found)
    kept = [p for p in parts if len(p) > 1]
    loose: list[int] = []
    for p in parts:
        if len(p) > 1:
            continue
        i = p[0]
        ties = [(sum(g.get_edge_data(i, j, {}).get("weight", 0.0) for j in c), k) for k, c in enumerate(kept)]
        best = max(ties, default=(0.0, -1))
        if best[0] > 0 and len(kept[best[1]]) < LEAF_FILES:
            kept[best[1]].append(i)
        else:
            loose.append(i)
    loose.sort()
    kept += [loose[k : k + LEAF_FILES] for k in range(0, len(loose), LEAF_FILES)]
    return [[paths[i] for i in sorted(p)] for p in kept if p]


# -- series ------------------------------------------------------------------------------------------------------------


def series(paths: list[str]) -> tuple[list[str], list[str]]:
    """The files of ``paths`` named by its most common numbered template, in the order of their numbers, and the
    rest; no series (all rest) unless SERIES_SHARE of the files share that template. Monthly bulletins, dated
    statements and numbered invoices read alike, so content puts 1969_05 beside 1981_05; their numbers keep a period
    together, and a question about a period can descend to it."""
    names = {p: p.rsplit("/", 1)[-1] for p in paths}
    templates = Counter(_DIGITS.sub("#", n) for n in names.values() if _DIGITS.search(n))
    if not templates:
        return [], paths
    template, count = templates.most_common(1)[0]
    if count < SERIES_SHARE * len(paths):
        return [], paths
    members = [p for p in paths if _DIGITS.sub("#", names[p]) == template]
    members.sort(key=lambda p: [int(d) for d in _DIGITS.findall(names[p])])
    rest = [p for p in paths if _DIGITS.sub("#", names[p]) != template]
    return members, rest


def span(first: str, last: str) -> str:
    """``first … last`` of a run of a series, the last without what it shares with the first."""
    a, b = first.rsplit("/", 1)[-1].rsplit(".", 1)[0], last.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    if a == b:
        return a
    same = len(next((a[:i] for i in range(len(a), 0, -1) if b.startswith(a[:i]) and not a[i - 1].isdigit()), ""))
    return f"{a} … {b[same:]}"


# -- the tree ----------------------------------------------------------------------------------------------------------


def build(paths: list[str], adj: Affinity) -> dict[str, Node]:
    """The tree over ``paths`` as nodes by key; the root's key is ``.``."""
    import numpy as np

    ids = itertools.count()
    nodes: dict[str, Node] = {}

    def add(kind: Literal["leaf", "inner"], label: str, files: list[str], children: list[str] = []) -> str:  # noqa: B006
        key = label if label not in nodes else f"{label} #{next(ids)}"
        nodes[key] = Node(key=key, label=label, kind=kind, files=sorted(files), children=children)
        return key

    def unit_label(prefix: str, names: list[str]) -> str:
        shown = ", ".join(names[:LABEL_NAMES]) + (f", +{len(names) - LABEL_NAMES}" if len(names) > LABEL_NAMES else "")
        return f"{prefix or '.'} · [{shown}]"

    def split_dir(prefix: str, files: list[str]) -> tuple[dict[str, list[str]], list[str]]:
        depth = len(prefix.split("/")) if prefix else 0
        subs: dict[str, list[str]] = defaultdict(list)
        loose: list[str] = []
        for p in files:
            segments = p.split("/")
            if (prefix and not p.startswith(prefix + "/")) or len(segments) <= depth + 1:
                loose.append(p)  # a direct file, or one pushed down from a parent level
            else:
                subs[segments[depth]].append(p)
        return subs, loose

    def merge_small(units: list[dict]) -> list[dict]:
        while len(units) > 1:
            small = sorted(
                (u for u in units if len(u["files"]) < LEAF_MIN), key=lambda u: (len(u["files"]), u["names"])
            )
            if not small:
                break
            u = small[0]
            at = units.index(u)
            fits = [v for v in units if v is not u and len(v["files"]) + len(u["files"]) <= LEAF_FILES]
            pool = fits or [v for v in units if v is not u]
            best = max(
                pool,
                key=lambda v: (
                    _aff(adj, u["files"], set(v["files"])) / math.sqrt(len(v["files"])),
                    -abs(units.index(v) - at),
                ),
            )
            if fits:
                merged = {
                    "type": "leaf",
                    "names": sorted(best["names"] + u["names"]),
                    "files": best["files"] + u["files"],
                }
                units = sorted([x for x in units if x is not u and x is not best] + [merged], key=lambda x: x["names"])
            else:  # every sibling is too big to merge with: the files go down into the most tied one
                best["files"] = best["files"] + u["files"]
                units = [x for x in units if x is not u]
        return units

    def bucket(units: list[dict]) -> list[dict]:
        """At most FAN_MAX children: agglomerate the most tied siblings (normalised by size) into clusters of at most
        FAN_MAX units; with no tie, the smallest neighbouring pair. Again when that still leaves too many."""
        if len(units) <= FAN_MAX:
            return units
        k = len(units)
        if all(u.get("series") for u in units):  # runs of a series: neighbours in order, a span each
            n = -(-k // FAN_MAX)
            out = []
            for at in range(0, k, n):
                us = units[at : at + n]
                if len(us) == 1:
                    out.append(us[0])
                    continue
                name = f"({span(us[0]['files'][0], us[-1]['files'][-1])})"
                files = [f for u in us for f in u["files"]]
                out.append({"type": "cluster", "names": [name], "files": files, "items": us, "series": True})
            return bucket(out) if len(out) > FAN_MAX else out
        owner = {f: i for i, u in enumerate(units) for f in u["files"]}
        m = np.zeros((k, k))
        for i, u in enumerate(units):
            for f in u["files"]:
                for g, w in adj.get(f, {}).items():
                    j = owner.get(g)
                    if j is not None and j != i:
                        m[i, j] += w
        size = np.array([len(u["files"]) for u in units], dtype=float)
        members = [[i] for i in range(k)]
        alive = list(range(k))
        while len(alive) > FAN_MAX:
            best = None
            for x, a in enumerate(alive):
                for b in alive[x + 1 :]:
                    if len(members[a]) + len(members[b]) > FAN_MAX:
                        continue
                    s = m[a, b] / math.sqrt(size[a] * size[b])
                    key = (s > 0, s, -(size[a] + size[b]), -abs(a - b))
                    if best is None or key > best[0]:
                        best = (key, a, b)
            if best is None:
                break
            _, a, b = best
            m[a, :] += m[b, :]
            m[:, a] += m[:, b]
            m[a, a] = 0
            size[a] += size[b]
            members[a] += members[b]
            alive.remove(b)
        out = []
        for a in alive:
            us = [units[i] for i in sorted(members[a])]
            if len(us) == 1:
                out.append(us[0])
            else:
                names = [n for u in us for n in u["names"]]
                out.append(
                    {"type": "cluster", "names": names, "files": [f for u in us for f in u["files"]], "items": us}
                )
        return bucket(out) if len(out) > FAN_MAX else out

    def build_unit(prefix: str, u: dict) -> str:
        if u["type"] == "sub":
            return make(u["path"], u["files"])
        if u["type"] == "leaf":
            plain = len(u["names"]) == 1 and not u["names"][0].startswith("(")
            return add("leaf", _join(prefix, u["names"][0]) if plain else unit_label(prefix, u["names"]), u["files"])
        kids = [build_unit(prefix, x) for x in u["items"]]
        return add("inner", unit_label(prefix, u["names"]), u["files"], kids)

    def make(prefix: str, files: list[str]) -> str:
        if len(files) <= LEAF_FILES:
            return add("leaf", prefix or ".", files)
        subs, loose = split_dir(prefix, files)
        if not loose and len(subs) == 1:
            ((name, sub),) = subs.items()
            return make(_join(prefix, name), sub)
        units: list[dict] = [
            {"type": "sub", "path": _join(prefix, n), "names": [n], "files": f} for n, f in sorted(subs.items())
        ]
        runs: list[dict] = []
        if len(loose) > LEAF_FILES:
            members, loose = series(sorted(loose))
            n = -(-len(members) // LEAF_FILES)
            for i in range(n):  # equal runs in number order
                run = members[i * len(members) // n : (i + 1) * len(members) // n]
                runs.append({"type": "leaf", "names": [f"({span(run[0], run[-1])})"], "files": run, "series": True})
        if loose:
            pieces = [sorted(loose)] if len(loose) <= LEAF_FILES else communities(sorted(loose), adj)
            units += [
                {
                    "type": "leaf",
                    "names": ["(files)" if len(pieces) == 1 else f"(files {i + 1}/{len(pieces)})"],
                    "files": pc,
                }
                for i, pc in enumerate(pieces)
            ]
        units = merge_small(units)
        if runs and units:  # a series beside other files and folders: the series is one child
            first, last = runs[0]["files"][0], runs[-1]["files"][-1]
            whole = {"type": "cluster", "names": [f"({span(first, last)})"], "items": bucket(runs), "series": True}
            units.append(runs[0] if len(runs) == 1 else whole | {"files": [f for r in runs for f in r["files"]]})
        elif runs:
            units = runs
        if len(units) == 1:
            return build_unit(prefix, units[0])
        return add("inner", prefix or ".", files, [build_unit(prefix, u) for u in bucket(units)])

    root = make("", sorted(paths)) if paths else add("leaf", ".", [])
    if root != ".":  # the root is always reachable by its key
        node = nodes.pop(root)
        nodes["."] = node.model_copy(update={"key": "."})
    return nodes


def _join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _unit(vector: csr_matrix) -> csr_matrix:
    norm = math.sqrt(vector.multiply(vector).sum())
    return vector / norm if norm else vector


# -- cards -----------------------------------------------------------------------------------------------------------


def write_cards(
    nodes: dict[str, Node],
    members: Mapping[str, str],
    names: Mapping[str, list[str]],
    centrality: Mapping[str, float],
    vectors: Mapping[str, csr_matrix],
    vocab: list[str],
    descriptions: Mapping[str, str],
    leads: Mapping[str, str],
) -> None:
    """Leaf: its keywords, then every member on one line, most central first. Inner: its keywords, each subgroup with
    its own keywords and its most central files with their top-level names, then the subtree's most central files.
    ``members`` is each file's member line, ``names`` its top-level names; keywords come from the TF-IDF ``vectors``
    (a node's vector is the balanced sum of its children's, so a large subgroup does not drown a small one; one-row
    sparse matrices, as dense ones were the vocabulary's size for every node).

    The root's card also says what the repository and each of its subgroups is for: it is the map the agent reads
    pinned in its context, and System 1 never judges it (the walk opens the root without asking). A purpose is
    extractive, never written by a model: a folder's own description (DESCRIBING, the first with one in
    ``descriptions``: a README's front matter or first paragraph, a package's description, an `__init__` docstring),
    or else the described folders under it that stand for it best (`typical`), a group with none standing for itself
    by its most central member's docstring (``leads``), never a document's first paragraph. Purposes on the cards
    System 1 judges were measured and left out: with lexical steering they added asks and no hits, and even the
    summary fixes behind them moved the walk; so descriptions live beside the summaries, unread by it."""
    from scipy.sparse import csr_matrix

    node_vectors: dict[str, csr_matrix] = {}

    def vector(node: Node) -> csr_matrix:
        if node.key not in node_vectors:
            parts = (
                [vector(nodes[k]) for k in node.children]
                if node.children
                else [vectors[p] for p in node.files if p in vectors]
            )
            units = [_unit(v) for v in parts]
            total = sum(units[1:], units[0]) if units else csr_matrix((1, len(vocab)))
            node_vectors[node.key] = _unit(total)
        return node_vectors[node.key]

    def keywords(node: Node, n: int) -> str:
        return ", ".join(grouping.top_terms(vector(node), vocab, n))

    def central(files: list[str]) -> list[str]:
        return sorted(files, key=lambda p: (-centrality.get(p, 0.0), p))

    def member(node: Node, p: str) -> str:
        folder = node.label.split(" · ")[0]
        name = p.removeprefix(folder + "/") if folder != "." else p
        return f"  {name} | {members[p]}" if members[p] else f"  {name}"

    found: dict[str, list[tuple[Node, str]]] = {}

    def own(node: Node) -> str:
        """The description of the one folder the node stands for; none for a group cut from several."""
        if " · " in node.label:
            return ""
        for name in DESCRIBING:
            path = name if node.label == "." else f"{node.label}/{name}"
            if _says(descriptions.get(path, "")):
                return descriptions[path]
        return ""

    def described(node: Node) -> list[tuple[Node, str]]:
        """(node, description) of the nearest described nodes at or under ``node``."""
        if node.key not in found:
            if text := own(node):
                found[node.key] = [(node, text)]
            elif node.kind == "leaf":
                lead = next((leads[p] for p in central(node.files) if _says(leads.get(p, ""))), "")
                found[node.key] = [(node, lead)] if lead else []
            else:
                found[node.key] = [d for k in node.children for d in described(nodes[k])]
        return found[node.key]

    def typical(entry: Node, of: Node) -> float:
        """How well a described node stands for ``of``: its vector's cosine with ``of``'s, times log(1 + its files).
        By size alone, Django's core package read as GeoDjango (contrib/gis is its largest described part)."""
        return float(vector(entry).multiply(vector(of)).sum()) * math.log1p(len(entry.files))

    def purpose(node: Node, limit: int) -> str:
        if text := own(node):
            return _clip(text, limit)
        entries = sorted(described(node), key=lambda d: (-typical(d[0], node), d[0].label))
        if node.kind == "leaf":
            return _clip(entries[0][1], limit) if entries else ""
        folder = node.label.split(" · ")[0]
        shown = [f"{_under(folder, entry.label)} — {_clip(text, ENTRY_CHARS)}" for entry, text in entries]
        return _clip("; ".join(shown), limit)

    for node in nodes.values():
        root = node.key == "."
        if node.kind == "leaf":
            lines = [f"group {node.label}  ({len(node.files)} files)"]
            if root and (text := own(node)):
                lines.append(f"purpose: {_clip(text, PURPOSE_CHARS)}")
            if words := keywords(node, cards.KEYWORDS):
                lines.append(f"keywords: {words}")
            lines.append("files:")
            lines += [member(node, p) for p in central(node.files)]
            node.card = cards._bounded(lines, cards.MEMBER_LINES_CHARS)
            continue
        kids = [nodes[k] for k in node.children]
        lines = [f"folder {node.label}  ({len(node.files)} files in {len(kids)} subgroups)"]
        if root and (text := purpose(node, PURPOSE_CHARS)):
            lines.append(f"purpose: {text}")
        if words := keywords(node, cards.KEYWORDS):
            lines.append(f"keywords: {words}")
        lines.append("subgroups:")
        per = max(200, (CARD_CHARS - 900) // max(1, len(kids)))
        for kid in kids:
            bits = []
            for p in central(kid.files)[:SUBGROUP_FILES]:
                tops = [n for n in names.get(p, []) if not n.startswith("_")][:TOP_NAMES]
                bits.append(p.rsplit("/", 1)[-1] + (" (" + ", ".join(tops) + ")" if tops else ""))
            words = keywords(kid, min(cards.KEYWORDS, cards.CHILD_KEYWORDS + len(kid.files) // 100))  # more for more
            said = purpose(kid, max(ENTRY_CHARS, per // 2)) if root else ""
            line = (
                f"  - {kid.label} ({len(kid.files)} files)"
                + (f": {said} |" if said else ":" if words else "")
                + (f" {words}" if words else "")
                + " | "
                + "; ".join(bits)
            )
            lines.append(line if len(line) <= per else line[: per - 3] + "...")
        lines.append("most central files:")
        lines += [member(node, p)[:200] for p in central(node.files)[:SUBTREE_FILES]]
        node.card = "\n".join(lines)[:CARD_CHARS]


def _says(summary: str) -> bool:
    return bool(summary) and not _PLACEHOLDER.search(summary)


def _under(folder: str, label: str) -> str:
    """A node's label as seen from ``folder``: `services/main_service` from `services` is `main_service`, and a group
    cut from ``folder`` itself (`ega · [a, b]` from `ega`) is `[a, b]`."""
    if folder == ".":
        return label
    if label.startswith(folder + "/"):
        return label[len(folder) + 1 :]
    return label.removeprefix(folder + " · ")


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
