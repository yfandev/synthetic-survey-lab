"""The look: paper, ink and the mascot's violet, in light and dark.

An off-white page, white cards, hairline rules, near-black ink, a warm gray for what is secondary; the mascot's
violet (`brand`) for what is Intern's own: the mark, the wordmark, the panels' tabs, the prompt's bar and cursor;
green only for what went right, amber for what needs a look, red for what went wrong. The mark is the mascot, the
violet cube in round glasses at the top of the README, drawn in half blocks (`widgets.mark`) in its own colours
(`MASCOT_COLOURS`), the same in both looks. The dark look swaps paper and ink, each a little grayed down, and
takes the mascot's lighter violet, which reads on it (`/dark`, `/light`).

Widgets name colours by role (`INK`, `S_MUTED`, ...), which are the theme's variables: `Content` reads them when
it draws, so a switch of theme redraws everything in the other look.
"""

from __future__ import annotations

from textual.theme import Theme

from intern import names

LIGHT = {
    "paper": "#fbfbf9",
    "card": "#ffffff",
    "wash": "#f4f4f0",
    "rule": "#e8e8e3",
    "ink": "#0c0c09",
    "ink-soft": "#2b2b22",
    "muted": "#7c7c67",
    "green": "#3fb55a",
    "amber": "#b7791f",
    "red": "#b23a3a",
    "select": "#dedcd3",
    "dim": "#d3d1c6",
    "brand": "#7c4ad2",
}
DARK = {
    "paper": "#141411",
    "card": "#1b1b17",
    "wash": "#20201b",
    "rule": "#2d2d26",
    "ink": "#e6e5de",
    "ink-soft": "#c4c3b8",
    "muted": "#8d8c78",
    "green": "#4cbf66",
    "amber": "#d4973a",
    "red": "#d9625f",
    "select": "#3a3a31",
    "dim": "#45443a",
    "brand": "#bd98ff",
}
"""`select`: text the mouse selected, a shade of the page, not ink at half strength. `brand`: the mascot's violet,
its deeper shade on the light page and its lavender on the dark one, each about as strong against the page."""
MASCOT_COLOURS = {
    "9": "#bb90ff",
    "8": "#ab76ff",
    "6": "#9257f2",
    "5": "#8650de",
    "4": "#7442c9",
    "3": "#6538b4",
    "K": "#1c1c1a",
    "W": "#ffffff",
}
"""The mascot's pixels (`widgets.MASCOT`): the cube's three faces, each a little darker lower down, the top `9` and
`8`, the left `6` and `5`, the right, in shadow, `4` and `3`; the glasses' frames, `K`; the eyes, `W`."""

PAPER = "$background"
CARD = "$surface"
WASH = "$wash"
RULE = "$rule"
INK = "$foreground"
INK_SOFT = "$primary"
MUTED = "$muted"
GREEN = "$success"
AMBER = "$warning"
RED = "$error"
SELECT = "$select"
BRAND = "$brand"
DIM = "$dim"
"""Between the rule and the muted gray: what System 1's chart draws that is not the turn's verdict."""
# Content styles, by role, so widgets never name a colour.
S_INK = INK
S_RULE = RULE
S_HEAD = f"bold {INK}"
S_MUTED = MUTED
S_HEAD_MUTED = f"bold {MUTED}"
S_MUTED_ITALIC = f"italic {MUTED}"
S_OK = GREEN
S_OK_BOLD = f"bold {GREEN}"
S_WARN = AMBER
S_WARN_BOLD = f"bold {AMBER}"
S_BAD = RED
S_BAD_BOLD = f"bold {RED}"


S_BOX = f"bold {INK} on {SELECT}"
"""The thinking level's box on the masthead, filled so a change of level is seen at once; `S_BOX_MUTED` its label."""
S_BOX_MUTED = f"{MUTED} on {SELECT}"

S_TAB = f"bold {PAPER} on {BRAND}"
"""A panel's name on its header, a tab in the mascot's violet."""

S_WORDMARK = f"bold {BRAND}"
WORDMARK = names.TITLE


def make_theme(name: str, c: dict[str, str], *, dark: bool) -> Theme:
    return Theme(
        name=name,
        primary=c["ink-soft"],
        secondary=c["muted"],
        accent=c["ink"],
        foreground=c["ink"],
        background=c["paper"],
        surface=c["card"],
        panel=c["wash"],
        success=c["green"],
        warning=c["amber"],
        error=c["red"],
        dark=dark,
        variables={
            "rule": c["rule"],
            "wash": c["wash"],
            "muted": c["muted"],
            "select": c["select"],
            "dim": c["dim"],
            "brand": c["brand"],
            "block-cursor-background": c["brand"],
            "block-cursor-foreground": c["paper"],
            "block-cursor-text-style": "none",
            "block-cursor-blurred-background": c["wash"],
            "block-cursor-blurred-foreground": c["ink"],
            "block-cursor-blurred-text-style": "none",
            "block-hover-background": c["wash"],
            "input-cursor-background": c["brand"],
            "input-cursor-foreground": c["paper"],
            "input-selection-background": c["rule"],
            "screen-selection-background": c["select"],
            "screen-selection-foreground": c["ink"],
            "border": c["rule"],
            "border-blurred": c["rule"],
            "scrollbar": c["wash"],
            "scrollbar-hover": c["rule"],
            "scrollbar-active": c["rule"],
            "scrollbar-background": c["paper"],
            "scrollbar-background-hover": c["paper"],
            "scrollbar-background-active": c["paper"],
            "link-color": c["ink"],
        },
    )


LIGHT_THEME = make_theme("intern-light", LIGHT, dark=False)
DARK_THEME = make_theme("intern-dark", DARK, dark=True)
