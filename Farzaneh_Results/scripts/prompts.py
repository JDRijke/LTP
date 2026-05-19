"""Prompt templates with fallacy definitions baked in (slide 10)."""

FALLACY_DEFINITIONS = {
    "authority": (
        "Appeal to Authority — claims something is true because an authority figure "
        "said so, even though that figure is not a legitimate expert on the matter, "
        "or the expertise is irrelevant to the claim."
    ),
    "black-white": (
        "Black-or-White (False Dilemma) — presents only two options as if they were "
        "the only possibilities, when in reality there are more."
    ),
    "hasty_generalization": (
        "Hasty Generalization — draws a broad conclusion from too few or unrepresentative examples."
    ),
    "natural": (
        "Appeal to Nature — argues that something is good, right, or better because it is 'natural', "
        "or that something is bad because it is 'unnatural'."
    ),
    "population": (
        "Appeal to Popularity (ad populum) — claims something is true or correct because "
        "many people believe it or do it."
    ),
    "slippery_slope": (
        "Slippery Slope — asserts that one step will inevitably lead to a chain of "
        "increasingly extreme consequences, without sufficient justification for the chain."
    ),
    "tradition": (
        "Appeal to Tradition — argues that something is right, good, or true simply "
        "because it has been done that way for a long time."
    ),
    "worse_problems": (
        "Appeal to Worse Problems (relative privation) — dismisses an issue by pointing to "
        "another, supposedly worse problem, rather than addressing the original issue on its merits."
    ),
}

SUBTASK1_SYSTEM = (
    "You are an expert in informal logic and argumentation theory. "
    "Your task is to decide whether an argument is fallacious or non-fallacious. "
    "A fallacious argument relies on faulty reasoning. A non-fallacious argument may "
    "use a recognizable reasoning pattern (e.g. citing an authority, citing tradition) "
    "in a legitimate way without committing a fallacy."
)

SUBTASK2_SYSTEM = (
    "You are an expert in informal logic and argumentation theory. "
    "You are given an argument that has already been identified as containing a fallacy. "
    "Your task is to identify which of the eight fallacy types it commits.\n\n"
    "Fallacy type definitions:\n"
    + "\n".join(f"- {k}: {v}" for k, v in FALLACY_DEFINITIONS.items())
)


def subtask1_prompt(text: str, *, cot: bool = False) -> str:
    if cot:
        return (
            f"Argument:\n\"\"\"\n{text}\n\"\"\"\n\n"
            "Think step by step about whether the argument's reasoning is sound. "
            "Then on the final line output exactly one of: `fallacy` or `non-fallacy`.\n"
            "Format the final line as: ANSWER: <label>"
        )
    return (
        f"Argument:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Answer with exactly one word: `fallacy` or `non-fallacy`.\n"
        "ANSWER:"
    )


def subtask2_prompt(text: str, *, cot: bool = False) -> str:
    label_list = ", ".join(FALLACY_DEFINITIONS.keys())
    if cot:
        return (
            f"Argument:\n\"\"\"\n{text}\n\"\"\"\n\n"
            "Think step by step about which fallacy pattern the argument follows. "
            f"Then on the final line output exactly one of: {label_list}.\n"
            "Format the final line as: ANSWER: <label>"
        )
    return (
        f"Argument:\n\"\"\"\n{text}\n\"\"\"\n\n"
        f"Choose the single best fallacy type from: {label_list}.\n"
        "ANSWER:"
    )


def build_fewshot_block(task: str, examples: list[dict]) -> str:
    """examples: [{text, label}] -- previously seen training examples."""
    parts = []
    for e in examples:
        parts.append(f"Argument: \"\"\"{e['text']}\"\"\"\nANSWER: {e['label']}")
    return "\n\n".join(parts)
