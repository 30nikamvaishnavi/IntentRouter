"""
IntentRouter Robustness Module
Generates synthetic perturbations for evaluating model fragility on real-world noise:
1. Keyboard typos & character swaps
2. All-lowercase with punctuation stripped
3. Conversational filler words & polite hedges
Marked as a small synthetic test.
"""

import random
import re
from typing import List, Dict, Any


KEYBOARD_ADJACENT = {
    'q': 'wa', 'w': 'qes', 'e': 'wrd', 'r': 'etf', 't': 'ryg', 'y': 'tuh',
    'u': 'yij', 'i': 'uok', 'o': 'ipl', 'p': 'ol',
    'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc', 'g': 'ftyhbv',
    'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm', 'l': 'kop',
    'z': 'asx', 'x': 'zsdc', 'c': 'xdfv', 'v': 'cfgb', 'b': 'vghn',
    'n': 'bhjm', 'm': 'njk'
}

FILLER_PREFIXES = [
    "umm hello there, sorry to bother you but basically ",
    "hey please can you help me quickly with this: ",
    "hi, so I have a quick question regarding my account, ",
    "excuse me, I was just wondering if someone could tell me ",
    "umm hey, ",
    "look, basically "
]


def perturb_typos(text: str, rng: random.Random) -> str:
    """Injects 1-2 realistic keyboard typos or character swaps into words."""
    words = text.split()
    if not words:
        return text

    # Select 1 or 2 words longer than 4 chars
    candidate_indices = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidate_indices:
        candidate_indices = list(range(len(words)))

    num_typos = min(len(candidate_indices), rng.choice([1, 2]))
    chosen_indices = rng.sample(candidate_indices, num_typos)

    new_words = list(words)
    for idx in chosen_indices:
        w = list(new_words[idx])
        action = rng.choice(["swap", "replace", "delete"])
        pos = rng.randint(1, max(1, len(w) - 2))

        if action == "swap" and len(w) > 2:
            w[pos], w[pos + 1] = w[pos + 1], w[pos]
        elif action == "replace":
            char_lower = w[pos].lower()
            if char_lower in KEYBOARD_ADJACENT:
                w[pos] = rng.choice(KEYBOARD_ADJACENT[char_lower])
        elif action == "delete" and len(w) > 3:
            del w[pos]

        new_words[idx] = "".join(w)

    return " ".join(new_words)


def perturb_lowercase_strip_punct(text: str) -> str:
    """Lowercases the text and strips all commas, periods, quotes, question marks."""
    lowered = text.lower()
    cleaned = re.sub(r'[^\w\s]', '', lowered)
    return re.sub(r'\s+', ' ', cleaned).strip()


def perturb_filler_words(text: str, rng: random.Random) -> str:
    """Prepends realistic conversational conversational filler hedges."""
    prefix = rng.choice(FILLER_PREFIXES)
    return prefix + text


def generate_perturbed_datasets(texts: List[str], seed: int = 42) -> Dict[str, List[str]]:
    """Generates all 3 perturbed variants for the provided dataset."""
    rng_typo = random.Random(seed)
    rng_filler = random.Random(seed + 1)

    return {
        "typos": [perturb_typos(t, rng_typo) for t in texts],
        "lowercase_strip": [perturb_lowercase_strip_punct(t) for t in texts],
        "filler_words": [perturb_filler_words(t, rng_filler) for t in texts]
    }


if __name__ == "__main__":
    sample = "Why is there an unexplained $4.50 charge on my recent transfer to France?"
    print(f"Original:        {sample}")
    print(f"Typos:           {perturb_typos(sample, random.Random(42))}")
    print(f"Lowercase/strip: {perturb_lowercase_strip_punct(sample)}")
    print(f"Filler words:    {perturb_filler_words(sample, random.Random(42))}")
