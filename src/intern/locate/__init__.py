"""The engine behind `decision_search`: a directory indexed as cards and chunks in a balanced search tree (`index`,
`processors`, `cards`, `grouping`, `tree`), searched by System 1 one card per state (`search`), kept fresh for an
extension's tool (`store`). Extensions choose the processors that build the index and the questions asked:
`extensions.code_search` for code, `extensions.workspace` for the files of an office. Needs numpy, scipy and networkx
(`intern-agent[code-search]`); nothing here imports an office library."""
