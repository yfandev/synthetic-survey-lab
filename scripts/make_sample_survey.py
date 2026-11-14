"""Build a sample questionnaire PDF shaped like a real GSS ballot.

Two pages, numbered questions with answer options, so NDI has something real to
read and the extract prompt can be tuned against a known answer key.

    python scripts/make_sample_survey.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

OUT = Path(__file__).resolve().parent.parent / "public" / "samples" / "sample_survey.pdf"

QUESTIONS = [
    (
        "Q1. Are we spending too much, too little, or about the right amount on education?",
        ["Spend too little", "About right", "Spend too much", "Don't know"],
    ),
    (
        "Q2. Taken all together, how would you say things are these days -- would you say that "
        "you are very happy, pretty happy, or not too happy?",
        ["Very happy", "Pretty happy", "Not too happy"],
    ),
    (
        "Q3. I'm going to read several things people might say about their jobs. How often do you "
        "find your work stressful?",
        ["Always", "Often", "Sometimes", "Hardly ever", "Never"],
    ),
    (
        "Q4. Would you say that most people can be trusted, or that you can't be too careful in "
        "dealing with people?",
        ["Most people can be trusted", "You can't be too careful", "Depends"],
    ),
    (
        "Q5. Do you think the government should do more to solve our country's problems, or should "
        "the government stay out of it?",
        ["Government should do more", "Government should stay out", "Agree with both"],
    ),
    (
        "Q6. How often do you attend religious services?",
        ["Never", "Less than once a year", "Once a year", "Several times a year", "Once a month", "Weekly or more"],
    ),
    (
        "Q7. Would you be willing to pay higher taxes in order to improve the quality of the "
        "environment?",
        ["Willing", "Not willing", "Depends"],
    ),
    (
        "Q8. On a scale of 1 to 7, where 1 is a strong Democrat and 7 is a strong Republican, where "
        "would you place yourself?",
        ["1 Strong Democrat", "2", "3", "4 Independent", "5", "6", "7 Strong Republican"],
    ),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=LETTER)
    width, height = LETTER

    for page, chunk in enumerate((QUESTIONS[:4], QUESTIONS[4:]), start=1):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(60, height - 70, "NATIONAL OPINION SURVEY")
        c.setFont("Helvetica", 9)
        c.drawString(60, height - 88, "Interviewer-administered ballot · confidential · page %d of 2" % page)
        c.line(60, height - 96, width - 60, height - 96)

        y = height - 124
        for text, options in chunk:
            c.setFont("Helvetica-Bold", 11)
            for line in _wrap(text, 78):
                c.drawString(60, y, line)
                y -= 15
            c.setFont("Helvetica", 10)
            for i, opt in enumerate(options, start=1):
                c.drawString(78, y, f"[{i}] {opt}")
                y -= 13
            y -= 12

        c.setFont("Helvetica-Oblique", 8)
        c.drawString(60, 70, "Please hand the completed ballot to the interviewer.")
        c.showPage()

    c.save()
    print(f"wrote {OUT} ({len(QUESTIONS)} questions, 2 pages)")


def _wrap(text: str, size: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > size:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


if __name__ == "__main__":
    main()
