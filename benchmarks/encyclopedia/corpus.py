"""The encyclopedia's facts (cadence#311): subject, relation and object triples read from
Wikidata (CC0 1.0) through a public SPARQL endpoint, each fact one sentence
"the {relation} of {subject} is {object} .".

``fetch`` runs the queries below and keeps each answer as it came, with its SHA-256, the
endpoint and the dump's date in ``manifest.json``. ``build`` turns the answers into one table of
facts:

- **Subjects.** Sovereign states (their capital, continent, currency and language), people with
  at least ``SITELINKS`` Wikipedia articles (their citizenship, occupation, birthplace, native
  language, sport and instrument) and cities with as many (their country). A subject's name is
  its English label, folded to lower-case ASCII words; a name two subjects share is dropped
  with both.
- **One object a relation.** Where Wikidata lists several answers, the one the most facts of
  that relation name, then the lowest item number.
- **The answers.** Each relation's ``TOP`` most named objects, each named by at least
  ``ANSWER_FACTS`` facts, make the closed answer vocabulary (an instrument's choice); facts of
  other objects are dropped (a capital is kept where another relation makes it an answer).
  Every answer is said as one word, its folded label joined by ``_``, and a subject that is an
  answer is said as that word.
- **Order.** Subjects by their articles, most first, so each size's facts are the best-known
  subjects' (the 10,000 facts are the first 10,000).

    python benchmarks/encyclopedia/corpus.py fetch --out raw
    python benchmarks/encyclopedia/corpus.py build --raw raw --out facts.tsv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ENDPOINT = "https://qlever.dev/api/wikidata"
AGENT = "cadence-encyclopedia-benchmark/0.98 (research benchmark)"
SITELINKS = 4
ANSWER_FACTS = 4
# the answers each relation gives at most: its most named objects
TOP = {"capital": 0, "continent": 10, "currency": 10, "language": 10, "citizenship": 250,
       "occupation": 100, "birthplace": 200, "tongue": 40, "sport": 40, "instrument": 25,
       "country": 250}
PREFIX = (
    "PREFIX wd: <http://www.wikidata.org/entity/> "
    "PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
    "PREFIX wikibase: <http://wikiba.se/ontology#> "
    "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
)
STATE = "wd:Q3624078"  # sovereign state
# relation word, subject pattern, property, object pattern ("" for any object)
RELATIONS = {
    "capital": ("state", "P36", ""),
    "continent": ("state", "P30", ""),
    "currency": ("state", "P38", ""),
    "language": ("state", "P37", ""),
    "citizenship": ("person", "P27", f"?o wdt:P31 {STATE} ."),
    "occupation": ("person", "P106", ""),
    "birthplace": ("person", "P19", ""),
    "tongue": ("person", "P103", ""),
    "sport": ("person", "P641", ""),
    "instrument": ("person", "P1303", ""),
    "country": ("city", "P17", f"?o wdt:P31 {STATE} ."),
}
SUBJECTS = {
    "state": f"?s wdt:P31 {STATE} . OPTIONAL {{ ?s wikibase:sitelinks ?n . }}",
    "person": f"?s wdt:P31 wd:Q5 . ?s wikibase:sitelinks ?n . FILTER(?n >= {SITELINKS})",
    "city": f"?s wdt:P31 wd:Q515 . ?s wikibase:sitelinks ?n . FILTER(?n >= {SITELINKS})",
}
# the relation's words in a sentence (``tongue`` is a person's native language)
SAID = {"tongue": "native language"}


def query(relation: str) -> str:
    kind, prop, obj = RELATIONS[relation]
    return (PREFIX + "SELECT ?s ?sl ?n ?o ?ol WHERE { " + SUBJECTS[kind] + f" ?s wdt:{prop} ?o . "
            + obj + ' ?s rdfs:label ?sl . FILTER(LANG(?sl) = "en") '
            + '?o rdfs:label ?ol . FILTER(LANG(?ol) = "en") }')


def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"endpoint": ENDPOINT, "agent": AGENT, "licence": "CC0 1.0",
                                   "queries": {}}
    dump = PREFIX + "SELECT ?v WHERE { wikibase:Dump ?p ?v } LIMIT 20"
    manifest["dump"] = [r[0] for r in rows(get(dump))]
    for relation in RELATIONS:
        text = query(relation)
        t0 = time.time()
        body = get(text)
        path = out / f"{relation}.tsv.gz"
        path.write_bytes(gzip.compress(body, compresslevel=6, mtime=0))
        manifest["queries"][relation] = {  # type: ignore[index]
            "query": text, "rows": body.count(b"\n") - 1,
            "sha256": hashlib.sha256(body).hexdigest(), "seconds": round(time.time() - t0, 1)}
        print(relation, manifest["queries"][relation]["rows"], flush=True)  # type: ignore[index]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def get(text: str) -> bytes:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": text, "send": "100000000"})
    request = urllib.request.Request(url, headers={"Accept": "text/tab-separated-values",
                                                   "User-Agent": AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return bytes(response.read())
        except Exception as error:  # a busy endpoint: wait and ask again
            print("retry", attempt, error, flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError("the endpoint did not answer")


def rows(body: bytes) -> list[list[str]]:
    lines = body.decode("utf-8").splitlines()[1:]
    return [line.split("\t") for line in lines if line]


def item(cell: str) -> int:
    return int(cell.rsplit("/Q", 1)[1].rstrip(">"))


def label(cell: str) -> str:
    text = cell[1: cell.rindex('"')] if cell.startswith('"') else cell
    return text.encode("utf-8").decode("unicode_escape", "ignore") if "\\u" in text else text


def fold(text: str) -> list[str]:
    """A label as lower-case ASCII words."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", plain.lower())


def build(raw: Path, out: Path) -> None:
    names: dict[int, list[str]] = {}
    known: dict[int, int] = {}
    triples: dict[str, dict[int, set[int]]] = {}
    object_names: dict[int, list[str]] = {}
    for relation in RELATIONS:
        with gzip.open(raw / f"{relation}.tsv.gz", "rb") as source:
            got: dict[int, set[int]] = defaultdict(set)
            for cells in rows(source.read()):
                if len(cells) < 5 or "/Q" not in cells[0] or "/Q" not in cells[3]:
                    continue
                s, o = item(cells[0]), item(cells[3])
                n = re.match(r'"?(\d+)', cells[2])
                known[s] = max(known.get(s, 0), int(n.group(1)) if n else 0)
                names[s] = fold(label(cells[1]))
                object_names[o] = fold(label(cells[4]))
                got[s].add(o)
            triples[relation] = got
    # a subject's name: its folded label, kept where no other subject shares it
    owners: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for s, words in names.items():
        if 0 < len(words) <= 6:
            owners[tuple(words)].add(s)
    named = {next(iter(v)): k for k, v in owners.items() if len(v) == 1}
    # the answers: each relation's TOP most named objects named by at least ANSWER_FACTS facts,
    # each said as one word (its folded label joined by ``_``) that no other object shares
    often = {r: Counter(o for objects in got.values() for o in objects)
             for r, got in triples.items()}
    words = {o: "_".join(w) for o, w in object_names.items() if w}
    answers: set[int] = set()
    for relation, counts in often.items():
        top = [o for o, n in counts.most_common() if n >= ANSWER_FACTS and o in words]
        answers.update(top[: TOP[relation]])
    spoken = Counter(words[o] for o in answers)
    answers = {o for o in answers if spoken[words[o]] == 1}
    said = {words[o] for o in answers}
    # one object a relation: of the subject's objects among the answers, the one the relation
    # names most, then the lowest item number
    facts: list[tuple[int, str, int]] = []
    for relation, got in triples.items():
        for s, objects in got.items():
            held = [o for o in objects if o in answers]
            if s in named and held:
                facts.append((s, relation, min(held, key=lambda o: (-often[relation][o], o))))
    facts.sort(key=lambda f: (-known.get(f[0], 0), f[0], list(RELATIONS).index(f[1])))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as sink:
        writer = csv.writer(sink, delimiter="\t", lineterminator="\n")
        writer.writerow(["subject", "name", "articles", "relation", "object", "answer"])
        for s, relation, o in facts:
            name = "_".join(named[s])  # a subject that is an answer is said as that one word
            writer.writerow([f"Q{s}", name if name in said else " ".join(named[s]),
                             known.get(s, 0), SAID.get(relation, relation), f"Q{o}", words[o]])
    kept = facts
    data = out.read_bytes()
    summary = {"facts": len(kept), "subjects": len({f[0] for f in kept}),
               "answers": len({f[2] for f in kept}),
               "relations": dict(Counter(f[1] for f in kept)),
               "sha256": hashlib.sha256(data).hexdigest(), "sitelinks": SITELINKS,
               "answer_facts": ANSWER_FACTS}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        fetch(args.out)
    else:
        build(args.raw, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
