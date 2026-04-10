"""Text normalization helpers ported from cleanup_dedup_articles.py."""


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")


def collapse_spaces(s: str) -> str:
    return " ".join(s.split())


def normalize_paragraphs(s: str, keep_blank_lines: int = 1) -> str:
    """Limit consecutive blank lines, strip trailing whitespace per line."""
    lines = s.split("\n")
    out = []
    blank_run = 0
    for line in lines:
        line = line.rstrip(" \t")
        if line.strip(" \t") == "":
            blank_run += 1
            if blank_run <= keep_blank_lines:
                out.append("")
        else:
            blank_run = 0
            out.append(line)
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def unwrap_soft_linebreaks(s: str) -> str:
    """Join wrapped lines into proper paragraphs, separated by double newlines."""
    lines = s.split("\n")
    paragraphs = []
    buf = []

    def flush():
        if not buf:
            return
        para = " ".join(part.strip() for part in buf if part.strip() != "")
        para = collapse_spaces(para)
        if para:
            paragraphs.append(para)
        buf.clear()

    for line in lines:
        if line.strip(" \t") == "":
            flush()
            paragraphs.append("")
        else:
            buf.append(line)
    flush()

    out = []
    prev_blank = True
    for p in paragraphs:
        if p == "":
            if not prev_blank:
                out.append("")
            prev_blank = True
        else:
            out.append(p)
            prev_blank = False
    while out and out[0] == "":
        out.pop(0)
    while out and out[-1] == "":
        out.pop()
    return "\n\n".join(out)


def clean_text(text: str) -> str:
    """Full text normalization pipeline."""
    text = normalize_newlines(text).replace("\t", " ")
    text = normalize_paragraphs(text, keep_blank_lines=1)
    text = unwrap_soft_linebreaks(text).strip()
    return text


def clean_article(article: dict) -> dict:
    """Clean text and title fields of an article in-place."""
    article["text"] = clean_text(article.get("text") or "")
    article["title"] = collapse_spaces(normalize_newlines(article.get("title") or "")).strip()
    return article


def is_valid(article: dict, min_text_length: int = 100, min_word_count: int = 20) -> bool:
    """Quality filter: enforce min text length, word count, and non-empty title."""
    text = article.get("text", "")
    if len(text) < min_text_length:
        return False
    if len(text.split()) < min_word_count:
        return False
    if not article.get("title", "").strip():
        return False
    return True
