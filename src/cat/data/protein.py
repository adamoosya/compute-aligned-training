"""Seeded synthetic complementary-hydrophobicity tasks (Appendix M)."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json
import random
from cat.rewards.protein import AMINO_ACIDS, HYDROPHOBIC, hydrophobicity


@dataclass(frozen=True)
class ProteinExample:
    example_id: str
    sequence: str
    target: str
    split: str

    @property
    def input_h(self) -> float:
        return hydrophobicity(self.sequence)

    @property
    def prompt(self) -> str:
        return f"Input: {self.sequence} \n Output:"


def make_protein_examples(count: int, *, seed: int = 42, split: str = "train") -> tuple[ProteinExample, ...]:
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("count must be positive and seed nonnegative integers")
    if split not in {"warmup", "train", "test"}:
        raise ValueError("split must be warmup, train or test")
    # Stable, separate streams; never reset evaluation to the training stream.
    stream = int.from_bytes(hashlib.sha256(f"cat-protein-v1:{seed}:{split}".encode()).digest()[:8], "big")
    rng = random.Random(stream)
    hydro, polar = sorted(HYDROPHOBIC), sorted(set(AMINO_ACIDS) - HYDROPHOBIC)
    result, seen = [], set()
    while len(result) < count:
        seq = "".join(rng.choices(AMINO_ACIDS, k=rng.randrange(10, 20)))
        if seq in seen:
            continue
        seen.add(seq)
        length = rng.randrange(10, 20)
        n = round((1 - hydrophobicity(seq)) * length)
        chars = rng.choices(hydro, k=n) + rng.choices(polar, k=length - n)
        rng.shuffle(chars)
        result.append(ProteinExample(f"{split}:{len(result)}", seq, "".join(chars), split))
    return tuple(result)


def protein_manifest(examples: tuple[ProteinExample, ...], *, seed: int) -> dict:
    records = [asdict(ex) for ex in examples]
    return {"generator": "cat-protein-v1", "seed": seed, "count": len(records),
            "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
            "records": records}
