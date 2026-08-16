import string

JOB_ID_NUMBER_WIDTH = 4


def candidate_prefixes(company_name: str) -> list[str]:
    """Candidate 2-letter prefixes derived from the company name, tried in order on collision.

    Never changes the prefix length/format — always exactly 2 letters (e.g. 'ZA', 'AM').
    """
    letters = [c for c in company_name.upper() if c.isalpha()]
    words = [w for w in company_name.upper().split() if w and w[0].isalpha()]

    candidates = []
    if len(letters) >= 2:
        candidates.append(letters[0] + letters[1])
    if len(letters) >= 3:
        candidates.append(letters[0] + letters[2])
    if len(words) >= 2:
        candidates.append(words[0][0] + words[1][0])
    if len(letters) >= 4:
        candidates.append(letters[0] + letters[3])

    seen: set[str] = set()
    unique = []
    for cand in candidates:
        if len(cand) == 2 and cand not in seen:
            unique.append(cand)
            seen.add(cand)
    return unique


def next_free_prefix(company_name: str, used_prefixes: set[str]) -> str:
    """Pick an unused 2-letter prefix: try name-derived candidates first, then A-Z sweep."""
    for cand in candidate_prefixes(company_name):
        if cand not in used_prefixes:
            return cand
    for a in string.ascii_uppercase:
        for b in string.ascii_uppercase:
            cand = a + b
            if cand not in used_prefixes:
                return cand
    raise RuntimeError("Exhausted all 2-letter company prefixes")  # practically unreachable


def format_job_id(prefix: str, number: int) -> str:
    return f"{prefix}{number:0{JOB_ID_NUMBER_WIDTH}d}"
