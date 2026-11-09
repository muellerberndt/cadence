"""A lexicon of real word forms and its wordpieces, for ``pieces.py``.

The pieces come from byte-pair merges over the word list, each word weighed once, in the
``##`` form of a word's later pieces: ``kube ##rnet ##es``. The words' grammar is the chamber's and
carries no English meaning: only the forms and how they share pieces are real. ``TECH`` is a
coder's words, always in the list ahead of the ranked words. The ranked list is passed as a file,
one word a line, most frequent first.

    python benchmarks/vocabulary/lexicon.py --ranked google10k.txt --words 5000 --pieces 1000 \\
        --out lexicon-5000.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

TECH = """
api apis app apps array arrays async await auth authenticate authentication authorization backend
backup bash batch binary bit bitwise bool boolean branch breakpoint browser buffer bug bugs build
builder byte bytes cache caching callback class classes cli client clone cloud cluster code codebase
coder commit commits compile compiler component config configure console constant container
containers context cookie core coroutine cpu crash cron css csv cursor daemon dashboard data
database dataset debug debugger decorator default dependency deploy deployment deserialize dev
developer devops dictionary diff directory distributed dns docker dockerfile docs driver dropdown
encoding encrypt encryption endpoint enum env environment error errors exception executable export
extension fallback fetch field file filesystem firewall float flutter folder fork framework
frontend function functions generic generics getter git github gitlab golang gpu grafana graphql
hash hashmap header heap helm hook hooks host hostname http https ide immutable import index
inference inheritance init inline input instance integer interface interpreter iterator java
javascript jenkins json jwt kafka kernel key keyword kotlin kubernetes lambda latency layout lint
linter linux list listener localhost lock log logger logging login loop macos malloc markdown
merge metadata method methods microservice microservices middleware migration mock module modules
mutex mutable namespace neovim network nginx node nodejs null numpy oauth object observability
offset opcode operator optimizer orm output overflow package parser parameter payload pipeline
plugin pointer polymorphism postgres prefetch process processor production prometheus prompt
protobuf proxy pull pytest python queue react readme rebase recursion recursive redis refactor
regex registry release repo repository request response rest retry rollback router runtime rust
sandbox scala schema script sdk semaphore serialize server serverless session setter shader shell
snapshot socket sql stack staging stateful stateless struct subnet sudo swift syntax tensor
terraform thread threads timeout token tokenizer tokens traceback transaction transformer tuple
typescript ubuntu unicode unittest upstream url utf variable vector virtual vscode webhook
websocket webpack wifi workflow worktree yaml zip
""".split()


def read_ranked(path: Path) -> list[str]:
    words = []
    for line in path.read_text().splitlines():
        w = line.strip().lower()
        if len(w) >= 2 and w.isalpha() and w.isascii():
            words.append(w)
    return words


def merges(words: list[str], pieces: int, keep: tuple[str, ...] = ("##s",)) -> dict[str, list[str]]:
    """Each word as its pieces after merging the most frequent adjacent pair until the inventory
    holds ``pieces`` pieces (``keep`` pieces are in it from the start and are never merged away)."""
    split = {w: [w[0]] + ["##" + c for c in w[1:]] for w in words}
    inventory = {p for parts in split.values() for p in parts} | set(keep)
    while len(inventory) < pieces:
        pairs: Counter[tuple[str, str]] = Counter()
        for parts in split.values():
            for a, b in zip(parts, parts[1:], strict=False):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        joined = a + b.removeprefix("##")
        for w, parts in split.items():
            if a not in parts:
                continue
            out, i = [], 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                    out.append(joined)
                    i += 2
                else:
                    out.append(parts[i])
                    i += 1
            split[w] = out
        inventory = {p for parts in split.values() for p in parts} | set(keep)
    return split


def build(ranked: list[str], words: int, pieces: int, longest: int = 3) -> dict[str, object]:
    """The first ``words`` forms of TECH and the ranked list that take at most ``longest`` pieces
    and do not end in the plural piece. Merges run on a list some way longer than the lexicon,
    so that a word's pieces are shared the way a larger vocabulary shares them."""
    pool = list(dict.fromkeys(TECH + ranked))[:int(words * 1.5) + 200]
    split = merges(pool, pieces)
    chosen = [w for w in pool if len(split[w]) <= longest and split[w][-1] != "##s"][:words]
    used = sorted({p for w in chosen for p in split[w]} | {"##s"})
    lengths = Counter(len(split[w]) for w in chosen)
    return {"words": chosen, "pieces": used, "split": {w: split[w] for w in chosen},
            "lengths": {str(k): v for k, v in sorted(lengths.items())},
            "tech": sum(1 for w in chosen if w in set(TECH))}


def whole(ranked: list[str], words: int) -> dict[str, object]:
    """The same forms with every word its own piece: one motor neuron per word, by the wordpiece
    language's grammar. A novel stem then has no taught piece at all (``untaught``)."""
    chosen = list(dict.fromkeys(TECH + ranked))[:words]
    return {"words": chosen, "pieces": [*chosen, "##s"], "split": {w: [w] for w in chosen},
            "lengths": {"1": len(chosen)}, "tech": sum(1 for w in chosen if w in set(TECH)),
            "untaught": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--ranked", type=Path, required=True)
    parser.add_argument("--words", type=int, default=200)
    parser.add_argument("--pieces", type=int, default=100)
    parser.add_argument("--longest", type=int, default=3)
    parser.add_argument("--whole", action="store_true", help="every word its own piece")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    ranked = read_ranked(args.ranked)
    lexicon = (whole(ranked, args.words) if args.whole
               else build(ranked, args.words, args.pieces, args.longest))
    args.out.write_text(json.dumps(lexicon))
    print(len(lexicon["words"]), "words", len(lexicon["pieces"]), "pieces", lexicon["lengths"],
          "tech", lexicon["tech"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
