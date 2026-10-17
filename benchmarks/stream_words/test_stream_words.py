"""The streaming spoken-word chamber: its lexicon, streams, labels, scorer and the controls'
interfaces."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("stream_words", HERE / "stream_words.py")
assert spec is not None and spec.loader is not None
chamber = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chamber)


@pytest.fixture(scope="module")
def protocol():
    return chamber.load_protocol()


@pytest.fixture(scope="module")
def lexicon(protocol):
    return chamber.make_lexicon(3, **chamber.lexicon_settings(protocol["world"]))


@pytest.fixture(scope="module")
def stream(protocol, lexicon):
    return chamber.make_stream(
        lexicon, 200, np.random.default_rng(7), **chamber.stream_settings(protocol["world"])
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 11])
def test_pairs_share_prefixes_and_rhymes_share_their_last_two_phones(protocol, seed):
    lex = chamber.make_lexicon(seed, **chamber.lexicon_settings(protocol["world"]))
    words, joint, pairs = lex["words"], lex["joint"], protocol["world"]["pairs"]
    assert len(words) == 2 * pairs and lex["phones"].shape == (16, 16)
    assert lex["phones"].min() >= 0 and lex["phones"].max() <= 1
    for k in range(pairs):
        a, b = words[2 * k], words[2 * k + 1]
        assert a[:-1] == b[:-1] and a[-1] != b[-1]  # ambiguous until the last frame
        assert len(a) in (4, 5)
    for w, word in enumerate(words):
        assert word[-2] == joint and joint not in word[:-2]
        assert all(x != y for x, y in zip(word, word[1:], strict=False))
        rhyme = lex["rhyme"][w]
        assert words[rhyme][-1] == word[-1] and rhyme // 2 != w // 2
        assert words[rhyme][-3] != word[-3]  # the phone before the joint decides
        assert lex["rhyme"][rhyme] == w and lex["pair"][lex["pair"][w]] == w
        assert sum(other[-1] == word[-1] for other in words) == 2
        for other in words:  # no word is the beginning of another
            assert other == word or other[: len(word)] != word


def test_the_lexicon_is_fixed_by_its_seed(protocol):
    settings = chamber.lexicon_settings(protocol["world"])
    first, again = chamber.make_lexicon(5, **settings), chamber.make_lexicon(5, **settings)
    assert first["words"] == again["words"] and np.array_equal(first["phones"], again["phones"])
    other = chamber.make_lexicon(6, **settings)
    assert other["words"] != first["words"] or not np.array_equal(other["phones"], first["phones"])


def test_labels_are_blank_except_at_each_words_last_frame(lexicon, stream):
    labels, ends, phones = stream["labels"], stream["ends"], stream["phones"]
    assert len(labels) == len(phones) == len(stream["frames"])
    assert int(ends.sum()) == len(stream["said"]) == 200
    assert np.array_equal(np.flatnonzero(ends) + 1, stream["boundaries"])
    assert np.array_equal(labels[ends] - 1, stream["said"])
    assert np.all(labels[~ends] == chamber.BLANK)
    start = 0
    for word, end in zip(stream["said"], stream["boundaries"], strict=True):
        segment = phones[start:end]
        gap = int(np.sum(segment < 0))
        assert 1 <= gap <= 3 and np.all(segment[:gap] < 0)  # silence, then the word
        spoken = segment[gap:]
        assert 4 <= len(spoken) <= 9
        runs = [int(p) for i, p in enumerate(spoken) if i == 0 or p != spoken[i - 1]]
        assert runs == lexicon["words"][int(word)]
        assert len(spoken) == 1 or spoken[-2] != spoken[-1]  # the final phone lasts one frame
        lengths = np.diff(np.flatnonzero(np.r_[True, spoken[1:] != spoken[:-1], True]))
        assert lengths.max() <= 2 and lengths[-1] == 1
        start = end
    assert start == len(labels)


def test_frames_are_phone_vectors_with_gaussian_noise(lexicon, stream):
    phones, frames = stream["phones"], stream["frames"]
    clean = np.where(phones[:, None] >= 0, lexicon["phones"][np.maximum(phones, 0)], 0.0)
    noise = frames - clean
    assert abs(noise.std() - 0.1) < 0.01 and abs(noise.mean()) < 0.01
    assert np.allclose(noise[phones < 0], frames[phones < 0])  # silence is noise alone


def test_the_window_presses_the_last_frames_into_one_observation():
    frames = np.arange(5 * 2, dtype=float).reshape(5, 2) + 1
    pressed = chamber.window(frames, 3)
    assert pressed.shape == (5, 6)
    assert np.array_equal(pressed[0], [0, 0, 0, 0, 1, 2])  # zeros before the stream began
    assert np.array_equal(pressed[4], frames[2:5].reshape(-1))  # oldest first, current last
    assert all(np.array_equal(pressed[t, -2:], frames[t]) for t in range(5))


def test_the_scorer_counts_ends_emissions_and_confusions(lexicon, stream):
    labels, ends = stream["labels"], stream["ends"]
    words = int(ends.sum())
    perfect = chamber.score(labels, stream, lexicon)
    assert perfect["word_accuracy"] == 1 and perfect["precision"] == 1
    assert perfect["emissions_elsewhere"] == 0 and perfect["end_emission"] == 1
    silent = chamber.score(np.zeros_like(labels), stream, lexicon)
    assert silent["word_accuracy"] == 0 and silent["precision"] is None
    assert silent["confusions"]["blank"] == words and silent["emissions_elsewhere"] == 0
    early = np.zeros_like(labels)
    early[np.flatnonzero(ends) - 1] = labels[ends]  # one frame before every end
    shifted = chamber.score(early, stream, lexicon)
    assert shifted["word_accuracy"] == 0 and shifted["emissions_elsewhere_per_word"] == 1
    assert shifted["precision"] == 0
    confused = labels.copy()
    at = np.flatnonzero(ends)
    confused[at[0]] = lexicon["pair"][labels[at[0]] - 1] + 1
    confused[at[1]] = lexicon["rhyme"][labels[at[1]] - 1] + 1
    scored = chamber.score(confused, stream, lexicon)
    assert scored["confusions"]["pair"] == 1 and scored["confusions"]["rhyme"] == 1
    assert scored["word_accuracy"] == pytest.approx((words - 2) / words)
    with pytest.raises(ValueError):
        chamber.score(labels[:-1], stream, lexicon)


def test_the_controls_read_nothing_or_the_current_phone_alone(lexicon, stream):
    scored = chamber.controls(stream, lexicon, 3)
    assert scored["majority"]["word_accuracy"] == 0
    assert scored["majority"]["emissions_elsewhere"] == 0
    assert scored["uniform"]["emission_rate_elsewhere"] == pytest.approx(12 / 13, abs=0.03)
    # the same noise-free phone always gets the same answer, so no history enters
    classes = len(lexicon["words"]) + 1
    table: dict[int, np.ndarray] = {}
    for phone, label in zip(stream["phones"], stream["labels"], strict=True):
        table.setdefault(int(phone), np.zeros(classes, int))[label] += 1
    best = sum(int(table[f].max()) for f in set(int(p) for p in stream["phones"][stream["ends"]]))
    assert scored["phone"]["word_accuracy"] <= best / int(stream["ends"].sum())


def test_the_protocol_names_the_chambers_settings(protocol):
    world = protocol["world"]
    longest = 2 * (world["prefix_phones"][1] + 1) + 1
    assert protocol["window_frames"] == longest == 9
    assert set(protocol["arms"]) <= set(chamber.BRAIN_ARMS)
    assert protocol["arms"]["surprise"]["genes"] == {"learning_surprise": True}
    assert protocol["arms"]["every_frame"]["genes"] == {"learning_surprise": False}
    assert protocol["arms"]["window"]["genes"] == protocol["arms"]["surprise"]["genes"]
    assert set(protocol["brain"]["genes"]) == {"actor_eta", "actor_eta_bias", "actor_eta_critic"}
    assert all(v == 0 for v in protocol["brain"]["genes"].values())
    assert protocol["gru"]["truncation"] >= longest + world["silence"][1]
    assert len(protocol["seeds"]) == 5


def test_gru_input_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    cell = chamber.competitors.GRUCell(4, 3, rng)
    x, h_prev, g = rng.normal(size=4), rng.normal(size=3) * 0.5, rng.normal(size=3)
    _, cache = cell.forward(x, h_prev)
    analytic = chamber.gru_input_gradient(cell, g, cache)
    eps = 1e-6
    for i in range(3):
        up, down = h_prev.copy(), h_prev.copy()
        up[i] += eps
        down[i] -= eps
        numeric = (g @ cell.forward(x, up)[0] - g @ cell.forward(x, down)[0]) / (2 * eps)
        assert analytic[i] == pytest.approx(numeric, abs=1e-8)


def test_truncated_backpropagation_is_the_gradient_of_the_frame_loss():
    """Through every cached frame of a window shorter than the truncation, back to its start."""
    learner = chamber.OnlineGRU(4, 5, 1, hidden=3, truncation=8)
    rng = np.random.default_rng(2)
    h0, xs, label = rng.normal(size=3) * 0.3, rng.normal(size=(4, 4)), 2

    def loss() -> float:
        h = h0
        for x in xs:
            h = learner.cell.forward(x, h)[0]
        logits = learner.W @ h + learner.b
        return float(np.log(np.sum(np.exp(logits - logits.max()))) + logits.max() - logits[label])

    learner.h = h0.copy()
    for x in xs:
        h, cache, p = learner._forward(x)
        learner.history.append(cache)
        learner.h = h
    grads = learner.gradients(label, p, h)
    eps = 1e-6
    for name, value in learner.params().items():
        for index in list(np.ndindex(value.shape))[:: max(1, value.size // 7)]:
            old = float(value[index])
            value[index] = old + eps
            up = loss()
            value[index] = old - eps
            down = loss()
            value[index] = old
            assert grads[name][index] == pytest.approx((up - down) / (2 * eps), abs=1e-7)


def test_the_online_gru_learns_a_delayed_label_and_its_frozen_copy_does_not():
    rng = np.random.default_rng(3)
    learner = chamber.OnlineGRU(3, 3, 0, hidden=8, truncation=4, rate=0.01)
    xs = np.eye(3)[rng.integers(3, size=3000)]
    labels = np.r_[0, np.argmax(xs[:-1], axis=1)]  # the label is the previous frame's class
    hits = [learner.learn(x, int(y)) == y for x, y in zip(xs, labels, strict=True)]
    assert np.mean(hits[-500:]) > 0.9 and learner.work["updates"] == 3000
    twin = learner.frozen_copy()
    before = {k: v.copy() for k, v in twin.params().items()}
    first = [twin.answer(x) for x in xs[:50]]
    assert all(np.array_equal(before[k], v) for k, v in twin.params().items())
    again = learner.frozen_copy()
    assert [again.answer(x) for x in xs[:50]] == first
    assert np.mean(np.array(first[1:]) == labels[1:50]) > 0.9


def test_a_brain_life_teaches_every_frame_and_its_copy_answers_alike(protocol, tmp_path):
    sizes = {"training_words": 6, "probe_every": 3, "evaluation_words": 4}
    life = chamber.brain_life("surprise", 0, protocol, tmp_path, sizes)
    assert life.get("refusal") is None and len(life["probes"]) == 2
    assert life["work"]["frames"] > 6 * 4 and 0 < life["work"]["lessons"] <= life["work"]["frames"]
    assert life["describe"]["genes"]["learning_surprise"] is True
    assert life["describe"]["genes"]["actor_eta"] == 0
    saved = tmp_path / "surprise-0-w6.npz"
    assert saved.exists() and not (tmp_path / "surprise-0-w3.npz").exists()
    test = chamber.make_stream(
        chamber.make_lexicon(0, **chamber.lexicon_settings(protocol["world"])),
        4,
        chamber.rng_for(0, "evaluation"),
        **chamber.stream_settings(protocol["world"]),
    )
    answers = chamber.brain_answers(saved, test["frames"])
    assert answers.tolist() == life["probes"][-1]["answers"]
    assert chamber.brain_answers(saved, test["frames"]).tolist() == answers.tolist()


def test_the_window_arm_composes_the_pressed_in_observation(protocol):
    brain = chamber.compose("window", 9 * 16, 0, protocol)
    assert brain.describe()["layout"]["inputs"] == 144
    assert brain.describe()["genes"]["learning_surprise"] is True
    stream = {"frames": np.zeros((3, 16))}
    assert chamber.observations("window", stream, protocol).shape == (3, 144)
    assert chamber.observations("surprise", stream, protocol).shape == (3, 16)


def test_a_smoke_life_writes_a_receipt_that_verifies_and_reports(tmp_path, capsys):
    assert chamber.main(["--arm", "gru", "--seed", "1", "--out", str(tmp_path), "--smoke"]) == 0
    receipt = tmp_path / "gru-1.json.gz"
    ok, _ = chamber.verify(receipt)
    assert ok
    body = chamber.read_receipt(receipt)["body"]
    assert body["status"] == "complete" and body["smoke"] is True
    assert body["result"]["parameters"] == 3 * (32 * 16 + 32 * 32 + 32) + 13 * 32 + 13
    assert set(body["result"]["controls"]) == set(chamber.CONTROLS)
    with pytest.raises(SystemExit):
        chamber.main(["--arm", "gru", "--seed", "1", "--out", str(tmp_path), "--smoke"])
    capsys.readouterr()
    assert chamber.main(["--report", str(tmp_path)]) == 0
    table = capsys.readouterr().out
    assert "| gru | 1 |" in table and "| majority | 1 |" in table
    assert json.loads(json.dumps(body["sizes"])) == {
        "training_words": 40,
        "probe_every": 20,
        "evaluation_words": 20,
    }


def test_the_history_amendment_adds_one_arm_and_changes_nothing_else(protocol):
    amended = chamber.load_protocol(HERE / "protocol-history.json")
    assert set(amended["arms"]) == set(protocol["arms"]) | {"history"}
    genes = amended["arms"]["history"]["genes"]
    assert genes["learning_surprise"] is True and genes["history_decays"] == [0.2, 0.8, 0.95]
    assert (genes["history_amplitude"], genes["history_scale"]) == (3.0, 1.2)
    for key, value in protocol.items():
        if key == "arms":
            assert all(amended["arms"][arm] == spec for arm, spec in value.items())
        elif key != "conditional_arm":
            assert amended[key] == value, key
    digest = __import__("hashlib").sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    assert digest in amended["amendment"]["amends"]
