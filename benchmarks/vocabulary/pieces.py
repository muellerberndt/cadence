"""The chamber's sentences said by wordpiece: one motor neuron per piece, not per word.

A word is a stem of pieces (``ref ##act ##or``) and a number piece: ``##0`` for the singular,
``##s`` for the plural of a noun. A verb's last piece agrees with its agent: ``##s`` with a singular
agent, ``##0`` with a plural one. ``##0`` is a silent piece, so that every form of a stem takes
one length and a batch's streams keep in step. ``near``, the brackets and ``end`` stay words.

Two ways of naming a word in a meaning (``route``):

- ``heard``: the meaning says the word's pieces, one moment each, every moment with its role flag
  (``AGENT``, ``PATIENT``, ``MOD``, ``VERB``; ``PLURAL`` on a plural noun), so the brain can say any
  word it hears whose pieces it knows. Input coordinates: the tokens and 11 flags.
- ``concept``: one moment per word, the word's own concept coordinate and flags, as in the
  grammar chamber: the brain has to know a word's pieces. Only taught words can be asked.

The grammar is the chamber's: ``clause`` (``agent (near mod)* verb patient end``), ``dyck`` and
``path`` (``X near Y end`` heard, the chain said), with ``swap`` beside them when ``swap`` is on:
the world says ``A verb B end``, the brain says ``B verb A end``, every noun used in the role it
was not heard in, and the verb agreeing with its new agent. Novel stems, a share of the stems per
founder, never appear in a training sentence and every piece of them does.

``pseudo`` is the share of the stems in a training sentence that are made up on the spot, one taught
first piece and taught later pieces drawn at random, the way a name or a new word arrives. Their
sentences teach the brain to say what it hears and leave no word to memorize. The ``pseudo`` panel
asks such stems at test.

``verb_tokens`` gives a verb's agreeing piece tokens of its own, ``##vs`` and ``##v0``. With the
shared pieces the read can only name a slot by what it holds, and a verb that agrees with a singular
agent says ``##s``, which the slots of plural nouns hold.

``start`` adds a ``START`` flag to the moment that hears a word's first piece: what the ``##`` of a
later piece says, as a coordinate all first pieces share, so an address can look for a word's start.

Forms and sharing of pieces are real (``lexicon.py``); the sentences carry no English meaning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from language import BATCH, SEEDS, TEST_HOPS, TRAIN_DEPTH, TRAIN_HOPS, TRAIN_L, Row, mix

Moment = tuple[int, ...]
NOVEL = 0.1  # the share of the stems held out as novel, per founder
LONG_L = (3, 4)


@dataclass(frozen=True)
class PieceSplit:
    """A founder's stems: taught and novel, noun and verb, by stem length."""

    nouns: dict[int, list[int]]  # stem length -> taught noun stems
    verbs: dict[int, list[int]]
    novel_nouns: dict[int, list[int]]
    novel_verbs: dict[int, list[int]]
    salt: int
    firsts: tuple[int, ...] = ()  # the first pieces of the taught stems
    laters: tuple[int, ...] = ()  # their later pieces

    def held(self, agent: int, patient: int) -> bool:
        return (mix(self.salt, 3, min(agent, patient), max(agent, patient)) % 6 == 0
                and agent != patient)


class PieceLanguage:
    """The chamber's families said by wordpiece, over a lexicon of real forms."""

    def __init__(self, lexicon: dict[str, Any] | str | Path, route: str = "heard",
                 swap: bool = False, families: dict[str, float] | None = None,
                 start: bool = False, pseudo: float = 0.0, verb_tokens: bool = False) -> None:
        if isinstance(lexicon, (str, Path)):
            lexicon = json.loads(Path(lexicon).read_text())
        self.route, self.swap, self.pseudo = route, swap, float(pseudo)
        self.untaught = bool(lexicon.get("untaught"))  # a novel stem may have no taught piece
        pieces = [p for p in lexicon["pieces"] if p not in ("##s", "##0")]
        self.piece_names = [*pieces, "##s", "##0", *(("##vs", "##v0") if verb_tokens else ())]
        self.piece_id = {p: i for i, p in enumerate(self.piece_names)}
        self.s, self.zero = self.piece_id["##s"], self.piece_id["##0"]
        self.verb_s, self.verb_zero = self.s, self.zero
        if verb_tokens:
            self.verb_s, self.verb_zero = self.piece_id["##vs"], self.piece_id["##v0"]
        self.tokens_pieces = len(self.piece_names)
        order = np.random.default_rng(np.random.SeedSequence([2026, 266])).permutation(
            len(lexicon["words"]))
        words = [lexicon["words"][int(i)] for i in order]
        self.word_names = words
        stems = [tuple(self.piece_id[p] for p in lexicon["split"][w]) for w in words]
        verbs = len(stems) // 3
        self.verb_stems, self.noun_stems = stems[:verbs], stems[verbs:]
        self.nouns, self.verbs = len(self.noun_stems), len(self.verb_stems)
        self.words = self.nouns + self.verbs  # stems: every stem has two forms
        P = self.tokens_pieces
        self.near = P
        self.open_round, self.close_round = P + 1, P + 2
        self.open_square, self.close_square = P + 3, P + 4
        self.end = P + 5
        self.actions = P + 6
        self.opens = (self.open_round, self.open_square)
        self.match = {self.open_round: self.close_round, self.open_square: self.close_square}
        f = self.actions
        self.plural, self.agent, self.patient = f, f + 1, f + 2
        self.mod, self.verb_flag = f + 3, f + 4
        self.c_open_round, self.c_open_square, self.c_close = f + 5, f + 6, f + 7
        self.c_from, self.go, self.speak, self.c_swap = f + 8, f + 9, f + 10, f + 11
        self.c_opens = (self.c_open_round, self.c_open_square)
        base = f + (12 if swap else 11)
        if not swap:
            self.c_swap = -1
        self.concept0 = base
        self.inputs = base + (len(stems) if route == "concept" else 0)
        self.start: int | None = None
        if start:
            self.start = self.inputs
            self.inputs += 1
        self.families = dict(families) if families else (
            {"clause": 0.4, "dyck": 0.2, "path": 0.2, "swap": 0.2} if swap
            else {"clause": 0.5, "dyck": 0.25, "path": 0.25})

    # ------------------------------------------------------------------------------ moments

    def dense(self, moments: list[Moment]) -> np.ndarray:
        x = np.zeros((len(moments), self.inputs))
        for row, moment in enumerate(moments):
            x[row, list(moment)] = 1.0
        return x

    def spoken(self, word: int) -> Moment:
        return (int(word), self.speak)

    def say(self, words: list[int]) -> str:
        names = {self.near: "near", self.open_round: "(", self.close_round: ")",
                 self.open_square: "[", self.close_square: "]", self.end: "end"}
        return " ".join(names.get(w, self.piece_names[w] if 0 <= w < self.tokens_pieces else "?")
                        for w in words)

    def noun_form(self, stem: Any, plural: int) -> list[int]:
        pieces = self.noun_stems[stem] if isinstance(stem, int) else stem
        return [*pieces, self.s if plural else self.zero]

    def verb_form(self, stem: Any, agent_plural: int) -> list[int]:
        pieces = self.verb_stems[stem] if isinstance(stem, int) else stem
        return [*pieces, self.verb_zero if agent_plural else self.verb_s]

    @staticmethod
    def made_up(rng: np.random.Generator, split: PieceSplit, length: int) -> tuple[int, ...]:
        """A stem of ``length`` taught pieces drawn at random."""
        rest = rng.choice(split.laters, size=length - 1) if length > 1 else ()
        return (int(rng.choice(split.firsts)), *(int(p) for p in rest))

    def noun_moments(self, stem: Any, plural: int, *flags: int) -> list[Moment]:
        number = (self.plural,) if plural else ()
        if self.route == "concept":
            return [(self.concept0 + self.verbs + stem, *number, *flags)]
        return self.pieces_heard(self.noun_form(stem, plural), *number, *flags)

    def verb_moments(self, stem: Any) -> list[Moment]:
        if self.route == "concept":
            return [(self.concept0 + stem, self.verb_flag)]
        pieces = self.verb_stems[stem] if isinstance(stem, int) else stem
        return self.pieces_heard(pieces, self.verb_flag)

    def pieces_heard(self, form: Any, *flags: int) -> list[Moment]:
        """A word's pieces as heard, one moment each, the first with ``START`` when it is on."""
        first = () if self.start is None else (self.start,)
        return [(p, *flags, *(first if i == 0 else ())) for i, p in enumerate(form)]

    def kind(self, r: Row, j: int) -> str:
        """What the ``j``-th word of a sentence is, for ``life.locate``."""
        for name, a, b in r.meta["slots"]:
            if a <= j < b:
                place = "number" if j == b - 1 else "first" if j == a else "later"
                return f"{r.family}_{name}_{place}"
        return f"{r.family}_grammar"

    # -------------------------------------------------------------------------------- split

    def split(self, founder: int) -> PieceSplit:
        """Stems by length; a tenth of them novel, every piece of a novel stem kept in a taught
        stem. ``##s`` and ``##0`` are taught by every sentence."""
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["split"]]))
        count: dict[int, int] = {}
        for stem in (*self.noun_stems, *self.verb_stems):
            for p in stem:
                count[p] = count.get(p, 0) + 1
        novel: set[tuple[str, int]] = set()
        everything = [("n", i) for i in range(self.nouns)] + [("v", i) for i in range(self.verbs)]
        target = int(NOVEL * len(everything))
        for k in rng.permutation(len(everything)):
            kind, i = everything[int(k)]
            stem = self.noun_stems[i] if kind == "n" else self.verb_stems[i]
            if len(novel) < target and (self.untaught
                                        or all(count[p] > stem.count(p) for p in stem)):
                novel.add((kind, i))
                for p in stem:
                    count[p] -= 1

        def buckets(kind: str, stems: list[tuple[int, ...]], held: bool) -> dict[int, list[int]]:
            out: dict[int, list[int]] = {}
            for i, stem in enumerate(stems):
                if ((kind, i) in novel) == held:
                    out.setdefault(len(stem), []).append(i)
            return out

        nouns, verbs = buckets("n", self.noun_stems, False), buckets("v", self.verb_stems, False)
        taught = [self.noun_stems[i] for b in nouns.values() for i in b]
        taught += [self.verb_stems[i] for b in verbs.values() for i in b]
        return PieceSplit(nouns, verbs,
                          buckets("n", self.noun_stems, True),
                          buckets("v", self.verb_stems, True),
                          int(rng.integers(2**31)),
                          tuple(sorted({stem[0] for stem in taught})),
                          tuple(sorted({p for stem in taught for p in stem[1:]})))

    @staticmethod
    def _length(rng: np.random.Generator, stems: dict[int, list[int]], low: int = 2) -> int:
        """A stem length, in proportion to the stems of that length, among lengths with enough
        stems for a batch's streams to differ."""
        lengths = [k for k, v in stems.items() if len(v) >= low] or list(stems)
        lengths.sort()
        weights = np.array([len(stems[k]) for k in lengths], dtype=float)
        return int(rng.choice(lengths, p=weights / weights.sum()))

    # ------------------------------------------------------------------------------ families

    def clause_shape(self, rng: np.random.Generator, split: PieceSplit, L: int,
                     novel: tuple[str, ...] = ()) -> dict[str, Any]:
        def pick(role: str, kind: str) -> int:
            taught = split.nouns if kind == "n" else split.verbs
            held = split.novel_nouns if kind == "n" else split.novel_verbs
            return self._length(rng, held if role in novel else taught)

        return {"agent": pick("agent", "n"), "patient": pick("patient", "n"),
                "verb": pick("verb", "v"),
                "mods": [pick("mod", "n") for _ in range(L)], "novel": novel,
                "inverted": bool(rng.integers(2)), "pseudo": 0.0}

    def clause(self, rng: np.random.Generator, split: PieceSplit, shape: dict[str, Any]) -> Row:
        novel, pseudo = shape["novel"], shape["pseudo"]

        def noun_of(role: str, length: int) -> tuple[Any, int]:
            if pseudo and rng.random() < pseudo:
                return self.made_up(rng, split, length), int(rng.integers(2))
            stems = split.novel_nouns if role in novel else split.nouns
            return int(rng.choice(stems[length])), int(rng.integers(2))

        while True:
            agent, patient = noun_of("agent", shape["agent"]), noun_of("patient", shape["patient"])
            if (novel or pseudo or not isinstance(agent[0], int) or not isinstance(patient[0], int)
                    or not split.held(agent[0], patient[0])):
                break
        verbs = split.novel_verbs if "verb" in novel else split.verbs
        v: Any = (self.made_up(rng, split, shape["verb"]) if pseudo and rng.random() < pseudo
                  else int(rng.choice(verbs[shape["verb"]])))
        mods = [noun_of("mod", k) for k in shape["mods"]]
        act = self.verb_moments(v)
        a = self.noun_moments(*agent, self.agent)
        p = self.noun_moments(*patient, self.patient)
        m = [self.noun_moments(*mod, self.mod) for mod in mods]
        heard: list[list[Moment]] = [act, p, a, *m] if shape["inverted"] else [act, a, *m, p]
        names = (["verb", "patient", "agent"] + ["mod"] * len(m) if shape["inverted"]
                 else ["verb", "agent"] + ["mod"] * len(m) + ["patient"])
        pre = [moment for part in heard for moment in part]
        where: dict[str, list[list[int]]] = {}
        at = 0
        for name, part in zip(names, heard, strict=True):
            where.setdefault(name, []).append(list(range(at, at + len(part))))
            at += len(part)
        target: list[int] = []
        slots: list[tuple[str, int, int]] = []
        src: list[int | None] = []

        def add(name: str, form: list[int], heard_at: list[int]) -> None:
            start = len(target)
            target.extend(form)
            slots.append((name, start, len(target)))
            for i in range(len(form)):
                src.append(heard_at[i] if i < len(heard_at) else
                           (heard_at[-1] if heard_at and self.route == "concept" else None))

        add("agent", self.noun_form(*agent), where["agent"][0])
        for k, mod in enumerate(mods):
            target.append(self.near)
            src.append(None)
            add("mod", self.noun_form(*mod), where["mod"][k])
        add("verb", self.verb_form(v, agent[1]), where["verb"][0])
        add("patient", self.noun_form(*patient), where["patient"][0])
        target.append(self.end)
        src.append(None)
        meta = {"agent": list(agent), "patient": list(patient), "verb": v, "L": len(mods),
                "mods": [list(x) for x in mods], "slots": slots, "sources": src,
                "inverted": shape["inverted"], "novel": list(novel),
                "attract": any(x[1] != agent[1] for x in mods)}
        return Row("clause", pre, target, meta)

    def swapped(self, rng: np.random.Generator, split: PieceSplit, shape: dict[str, Any]) -> Row:
        """The world says ``A verb B end`` by pieces with no flags; ``SWAP``; the brain says ``B
        verb A end``, the verb agreeing with B."""
        novel, pseudo = shape["novel"], shape["pseudo"]

        def noun_of(role: str, length: int) -> tuple[Any, int]:
            if pseudo and rng.random() < pseudo:
                return self.made_up(rng, split, length), int(rng.integers(2))
            stems = split.novel_nouns if role in novel else split.nouns
            return int(rng.choice(stems[length])), int(rng.integers(2))

        a, b = noun_of("agent", shape["agent"]), noun_of("patient", shape["patient"])
        verbs = split.novel_verbs if "verb" in novel else split.verbs
        v: Any = (self.made_up(rng, split, shape["verb"]) if pseudo and rng.random() < pseudo
                  else int(rng.choice(verbs[shape["verb"]])))
        heard_a, heard_b = self.noun_form(*a), self.noun_form(*b)
        heard_v = self.verb_form(v, a[1])
        said_v = self.verb_form(v, b[1])
        pre: list[Moment] = [*self.pieces_heard(heard_a), *self.pieces_heard(heard_v),
                             *self.pieces_heard(heard_b), (self.end,), (self.c_swap,)]
        at_a = list(range(len(heard_a)))
        at_v = list(range(len(heard_a), len(heard_a) + len(heard_v)))
        at_b = list(range(at_v[-1] + 1, at_v[-1] + 1 + len(heard_b)))
        target = [*heard_b, *said_v, *heard_a, self.end]
        slots = [("agent", 0, len(heard_b)), ("verb", len(heard_b), len(heard_b) + len(said_v)),
                 ("patient", len(heard_b) + len(said_v), len(target) - 1)]
        src: list[int | None] = [*at_b, *at_v[:-1], None, *at_a, None]
        meta = {"agent": list(b), "patient": list(a), "verb": v, "L": 0, "mods": [],
                "slots": slots, "sources": src, "inverted": False, "novel": list(novel),
                "attract": False, "swap": True}
        return Row("swap", pre, target, meta)

    def swap_shape(self, rng: np.random.Generator, split: PieceSplit,
                   novel: tuple[str, ...] = ()) -> dict[str, Any]:
        return self.clause_shape(rng, split, 0, novel)

    def dyck(self, rng: np.random.Generator, pattern: list[int]) -> Row:
        types = [int(rng.integers(2)) for _ in pattern]
        pre: list[Moment] = []
        words: list[int] = []
        src: list[int | None] = []
        stack: list[tuple[int, int]] = []
        for i, (step, t) in enumerate(zip(pattern, types, strict=True)):
            if step:
                pre.append((self.c_opens[t],))
                words.append(self.opens[t])
                stack.append((self.opens[t], i))
                src.append(i)
            else:
                pre.append((self.c_close,))
                o, j = stack.pop()
                words.append(self.match[o])
                src.append(j)
        depth, deepest = 0, 0
        for step in pattern:
            depth += 1 if step else -1
            deepest = max(deepest, depth)
        for o, j in reversed(stack):
            words.append(self.match[o])
            src.append(j)
        meta = {"prefix": len(pattern), "depth": deepest, "open": len(stack),
                "nested": 0 not in pattern, "sources": [*src, None]}
        return Row("dyck", pre, [*words, self.end], meta)

    def path(self, rng: np.random.Generator, split: PieceSplit, hops: int, length: int,
             novel: bool = False, distractors: int = 2, pseudo: float = 0.0) -> Row:
        """Facts heard by pieces; the chain from the named start said by pieces. Every entity of
        a row has a stem of ``length`` pieces. With ``novel``, the chain's stems are novel and the
        distractors' taught; with ``pseudo``, that share of the chain's stems is made up."""
        taught = split.nouns[length]
        held = split.novel_nouns.get(length, []) if novel else taught
        need = hops + 1
        if len(held) >= need:
            stems: list[Any] = [int(i) for i in rng.permutation(held)[:need]]
        else:
            stems = [int(i) for i in rng.permutation(taught)[:need]]
        stems = [self.made_up(rng, split, length) if pseudo and rng.random() < pseudo else s
                 for s in stems]
        chain = [(s, int(rng.integers(2))) for s in stems]
        rest = [s for s in taught if s not in set(stems)]
        firsts = [int(i) for i in rng.permutation(rest)[:distractors]]
        others = [int(i) for i in rng.permutation(rest)[:distractors * 2 + 2]]
        facts = [(chain[i], chain[i + 1]) for i in range(hops)]
        for k, f in enumerate(firsts):
            other = next(o for o in others[k:] if o != f)
            facts.append(((f, int(rng.integers(2))), (other, int(rng.integers(2)))))
        pre: list[Moment] = []
        spots: dict[tuple[int, int], list[int]] = {}
        for i in rng.permutation(len(facts)):
            x, y = facts[int(i)]
            fx, fy = self.noun_form(*x), self.noun_form(*y)
            at = len(pre)
            pre += [*self.pieces_heard(fx), (self.near,)]
            spots[y] = list(range(len(pre), len(pre) + len(fy)))
            pre += [*self.pieces_heard(fy), (self.end,)]
            spots.setdefault(x, list(range(at, at + len(fx))))
        start_at = len(pre)
        start = chain[0]
        pre += (self.pieces_heard(self.noun_form(*start), self.c_from) if self.route == "heard"
                else [(self.concept0 + self.verbs + int(start[0]),
                       *((self.plural,) if start[1] else ()), self.c_from)])
        words: list[int] = []
        src: list[int | None] = []
        slots: list[tuple[str, int, int]] = []
        for k, e in enumerate(chain):
            if k:
                words.append(self.near)
                src.append(None)
            form = self.noun_form(*e)
            slots.append(("start" if k == 0 else "hop", len(words), len(words) + len(form)))
            words += form
            if k == 0:
                src += [(start_at + i if self.route == "heard" else start_at)
                        for i in range(len(form))]
            else:
                src += list(spots[e])
        return Row("path", pre, [*words, self.end],
                   {"hops": hops, "chain": [list(e) for e in chain], "facts": len(facts),
                    "slots": slots, "sources": [*src, None], "novel": ["chain"] if novel else []})

    def train_batch(self, rng: np.random.Generator, split: PieceSplit, grammar: str = "g1",
                    size: int = BATCH) -> list[Row]:
        family = str(rng.choice(list(self.families), p=list(self.families.values())))
        if family == "clause":
            shape = self.clause_shape(rng, split, int(rng.choice(TRAIN_L)))
            shape["pseudo"] = self.pseudo
            return [self.clause(rng, split, shape) for _ in range(size)]
        if family == "swap":
            shape = self.swap_shape(rng, split)
            shape["pseudo"] = self.pseudo
            return [self.swapped(rng, split, shape) for _ in range(size)]
        if family == "dyck":
            pattern = dyck_pattern(rng, int(rng.integers(1, 7)), TRAIN_DEPTH)
            return [self.dyck(rng, pattern) for _ in range(size)]
        hops = int(rng.choice(TRAIN_HOPS))
        length = self._length(rng, split.nouns, low=8)
        return [self.path(rng, split, hops, length, pseudo=self.pseudo) for _ in range(size)]

    # -------------------------------------------------------------------------------- panels

    def panels(self, founder: int, batches: int = 4) -> dict[str, list[list[Row]]]:
        split = self.split(founder)
        rng = np.random.default_rng(np.random.SeedSequence([int(founder), SEEDS["test"]]))
        out: dict[str, list[list[Row]]] = {}

        def clauses(Ls: tuple[int, ...], novel: tuple[str, ...] = (),
                    pseudo: float = 0.0) -> list[list[Row]]:
            rows = []
            for i in range(batches):
                shape = self.clause_shape(rng, split, Ls[i % len(Ls)], novel)
                shape["pseudo"] = pseudo
                rows.append([self.clause(rng, split, shape) for _ in range(BATCH)])
            return rows

        out["taught"] = clauses(TRAIN_L)
        out["long"] = clauses(LONG_L)
        if self.route == "heard":
            out["novel_agent"] = clauses(TRAIN_L, ("agent",))
            out["novel_patient"] = clauses(TRAIN_L, ("patient",))
            out["novel_all"] = clauses(TRAIN_L, ("agent", "patient", "mod", "verb"))
        out["dyck"] = []
        for i in range(batches):
            nested = i % 2 == 0
            length = int(rng.integers(1 if nested else 3, 4 if nested else 7))
            pattern = dyck_pattern(rng, length, TRAIN_DEPTH, nested=nested)
            out["dyck"].append([self.dyck(rng, pattern) for _ in range(BATCH)])

        def paths(hops: tuple[int, ...], novel: bool = False) -> list[list[Row]]:
            rows = []
            for i in range(batches):
                stems = split.novel_nouns if novel else split.nouns
                need = max(hops) + 1 if novel else 8
                length = self._length(rng, {k: v for k, v in stems.items() if len(v) >= need}
                                      or stems, low=need)
                rows.append([self.path(rng, split, hops[i % len(hops)], length, novel)
                             for _ in range(BATCH)])
            return rows

        out["path"] = paths(TRAIN_HOPS)
        out["path_long"] = paths(TEST_HOPS)
        if self.route == "heard":
            out["path_novel"] = paths(TRAIN_HOPS, True)
        if self.pseudo:
            pseudo_rng = np.random.default_rng(np.random.SeedSequence([int(founder), 266, 5]))
            kept, rng = rng, pseudo_rng
            out["pseudo"] = clauses(TRAIN_L, (), 1.0)
            rng = kept
        if self.swap:
            out["swap"] = [[self.swapped(rng, split, shape) for _ in range(BATCH)]
                           for shape in (self.swap_shape(rng, split) for _ in range(batches))]
            out["swap_novel"] = [[self.swapped(rng, split, shape) for _ in range(BATCH)]
                                 for shape in (self.swap_shape(rng, split, ("agent", "patient"))
                                               for _ in range(batches))]
        return out

    # ------------------------------------------------------------------------------- scoring

    def score(self, row: Row, said: list[int]) -> dict[str, Any]:
        target = row.target[row.force:]
        out: dict[str, Any] = {"exact": said == target}
        ended = bool(said) and said[-1] == self.end
        if row.family in ("clause", "swap"):
            slots = row.meta["slots"]
            hit = {name: all(said[a:b] == target[a:b] for n, a, b in slots if n == name)
                   for name, _, _ in slots}
            out["agent_word"] = hit.get("agent")
            out["patient_word"] = hit.get("patient")
            out["verb_word"] = hit.get("verb")
            out["mod_words"] = hit.get("mod") if row.meta["L"] else None
            verb = next((a, b) for n, a, b in slots if n == "verb")
            out["agree"] = (len(said) >= verb[1] and said[verb[1] - 1] == target[verb[1] - 1])
            out["ended"] = ended
            out["first_piece"] = bool(said) and said[0] == target[0]
            out["words_right"] = float(np.mean([said[a:b] == target[a:b] for _, a, b in slots]))
        elif row.family == "dyck":
            stack: list[int] = []
            valid = ended
            for w in (said[:-1] if ended else said):
                if w in self.opens:
                    stack.append(w)
                elif w in (self.close_round, self.close_square) and stack and (
                        self.match[stack[-1]] == w):
                    stack.pop()
                else:
                    valid = False
            out["valid"] = valid and not stack and len(said) == len(target)
        elif row.family == "path":
            slots = row.meta["slots"]
            run = 0
            for _, a, b in slots:
                if said[a:b] != target[a:b]:
                    break
                run += 1
            out["hops"] = max(0, run - 1)
            out["first_hop"] = run >= 2
            out["words_right"] = float(np.mean([said[a:b] == target[a:b] for _, a, b in slots]))
        return out

    def tally(self, rows: list[Row], said: list[list[int]]) -> dict[str, Any]:
        scores = [self.score(r, s) for r, s in zip(rows, said, strict=True)]
        out: dict[str, Any] = {"n": len(scores)}
        for key in scores[0]:
            kept = [s[key] for s in scores if s.get(key) is not None]
            out[key] = float(np.mean(kept)) if kept else None
        return out

    @staticmethod
    def conditions(r: Row) -> list[str]:
        m = r.meta
        if r.family == "clause":
            return [f"L{m['L']}", "inverted" if m["inverted"] else "canonical"]
        if r.family == "dyck":
            return [f"depth{m['depth']}", "nested" if m["nested"] else "mixed"]
        if r.family == "path":
            return [f"hops{m['hops']}"]
        return []

    def read_speaker(self, speaker: Any, next_words: Any, batches: list[list[Row]], *,
                     twins: bool = False) -> dict[str, Any]:
        rows_all: list[Row] = []
        said_all: list[list[int]] = []
        hits: list[float] = []
        conds: dict[str, list[int]] = {}
        for rows in batches:
            rows_all += rows
            said_all += speaker(rows)
            if next_words is not None:
                hits += next_words(rows)
        out = self.tally(rows_all, said_all)
        if next_words is not None:
            out["next_word"] = float(np.mean(hits))
        for r, s in zip(rows_all, said_all, strict=True):
            for key in self.conditions(r):
                c = conds.setdefault(key, [0, 0])
                c[0] += int(s == r.target[r.force:])
                c[1] += 1
        out["conditions"] = {k: round(v[0] / v[1], 4) for k, v in sorted(conds.items())}
        out["examples"] = [[self.say(r.target[r.force:]), self.say(s)] for r, s in
                           list(zip(rows_all, said_all, strict=True))[:4]]
        return out


def dyck_pattern(rng: np.random.Generator, length: int, max_depth: int,
                 nested: bool | None = None) -> list[int]:
    from language import Language

    return Language.dyck_pattern(rng, length, max_depth, nested=nested)


PANELS = ("taught", "long", "novel_agent", "novel_patient", "novel_all", "pseudo", "dyck", "path",
          "path_long", "path_novel", "swap", "swap_novel")
