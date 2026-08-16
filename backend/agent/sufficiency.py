"""Deterministic "enough information" rule — never left to LLM judgment alone.

Hard floor (always required, checked in code):
  - job_title is set
  - at least one of required_skills / responsibilities is non-empty

Soft essentials (experience / location / work_mode "if relevant" to the role) are
context-dependent, so `analyze_turn`'s prompt asks the model to flag them in
`missing_essential` only when they genuinely matter for that job. `apply_updates`
combines both: sufficiency is reached once the hard floor holds AND the model
isn't currently flagging any soft-essential as missing.
"""

HARD_REQUIRED_SCALAR = ("job_title",)
HARD_REQUIRED_ANY_OF_LISTS = ("required_skills", "responsibilities")


def hard_floor_met(job_state: dict) -> bool:
    has_title = all(bool(job_state.get(f)) for f in HARD_REQUIRED_SCALAR)
    has_requirements = any(bool(job_state.get(f)) for f in HARD_REQUIRED_ANY_OF_LISTS)
    return has_title and has_requirements


def hard_floor_missing(job_state: dict) -> list[str]:
    missing = [f for f in HARD_REQUIRED_SCALAR if not job_state.get(f)]
    if not any(job_state.get(f) for f in HARD_REQUIRED_ANY_OF_LISTS):
        missing.append("required_skills_or_responsibilities")
    return missing


def sufficiency_ok(job_state: dict, llm_enough_information: bool, llm_missing_essential: list[str]) -> bool:
    if not hard_floor_met(job_state):
        return False
    return llm_enough_information and not llm_missing_essential


def combined_missing_essential(job_state: dict, llm_missing_essential: list[str]) -> list[str]:
    missing = hard_floor_missing(job_state)
    for field in llm_missing_essential:
        if field not in missing:
            missing.append(field)
    return missing
