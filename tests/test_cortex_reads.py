"""What the cortex of a brain with a fixed code reads (``compose(cortex_reads=...)``, 0.102.0):
the code (the founder), the code and the observation itself (``"both"``), or the observation
itself (``"observation"``). Every memory keys on the code whichever the cortex reads."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import cadence as cd
from cadence import brain as kernel
from cadence import generic
from cadence.genome import Genome, develop
from cadence.senses import Ear, Things

SENSES = {"eyes": {"hue": 12, "shape": 20}, "ear": 8}
X = {"eyes": {"hue": np.eye(12)[[3]], "shape": np.linspace(0.0, 1.0, 20)[None]},
     "ear": np.eye(8)[[5]]}


def compose(reads: str | None = None, **options) -> cd.Brain:
    named = {} if reads is None else {"cortex_reads": reads}
    settings = {"senses": SENSES, "actions": 3, "code": 64, "modules": (16, 16), "seed": 4,
                "backend": "cpu", "recall_capacity": 8, **options}
    return cd.Brain.compose(**settings, **named)


def life(brain: cd.Brain, moments: int = 6) -> list[int]:
    rng = np.random.default_rng(1)
    acts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(moments):
            x = {"eyes": {"hue": rng.random((1, 12)), "shape": rng.random((1, 20))},
                 "ear": np.eye(8)[[t % 8]]}
            brain.lesson(x, [t % 3])
            acts.append(int(brain.act(x, greedy=True)[0]))
    return acts


def saved(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["brain"]))


# -- the layout


def test_both_is_the_code_brain_plus_the_observation_and_its_projection():
    code, both = compose(), compose("both")
    old, new = code.connectome, both.connectome
    observed = both.connectome.populations["observation"]
    assert len(observed) == both.inputs == 40 and len(both.observation_index) == 40
    assert new.n == old.n + 40
    assert list(observed) == list(range(old.n, old.n + 40))  # appended last
    kept = ~np.isin(new.pre, observed)
    for name in ("pre", "post"):
        assert np.array_equal(getattr(new, name)[kept], getattr(old, name))
    first = set(new.populations["module_0"])
    assert set(new.post[~kept].tolist()) == first and not np.isin(new.post, observed).any()
    assert int((~kept).sum()) == 40 * 16  # one synapse a pair, into the first module
    assert np.array_equal(both.learner.brain.efficacy[kept], code.learner.brain.efficacy)
    assert np.array_equal(both.learner.brain.bias[: old.n], code.learner.brain.bias)
    assert both.parameters() == code.parameters() + 40 * 16 + 40
    # every memory is the code brain's
    assert both.code_width == code.code_width == 64
    for name in ("sensory", "history", "recall", "next"):
        assert len(new.populations[name]) == len(old.populations[name])
    assert np.array_equal(both.observation_code(X), code.observation_code(X))


def test_observation_holds_the_observation_in_the_sensory_region():
    brain = compose("observation")
    populations = brain.connectome.populations
    assert len(populations["sensory"]) == 40 and "observation" not in populations
    assert len(populations["history"]) == 40  # the history hears what the cortex reads
    assert brain.code_width == 64 and brain.recall.key_width == 64  # the memories: the code
    assert brain.observation_code(X).shape == (1, 64)


def test_the_drive_holds_the_code_in_the_sensory_region_and_the_observation_beside_it():
    brain = compose("both")
    row = np.asarray(brain._observations(X))
    drive = brain.stimulus(X, memory=False)
    amplitude = brain.brain.neuron_model.stimulus_amplitude
    sensory = brain.connectome.populations["sensory"]
    observed = brain.connectome.populations["observation"]
    assert np.allclose(drive[:, observed], amplitude * row)
    assert np.allclose(drive[:, sensory], amplitude * brain.observation_code(X))
    code = compose()
    assert np.allclose(code.stimulus(X, memory=False)[:, code.connectome.populations["sensory"]],
                       drive[:, sensory])
    alone = compose("observation")
    assert np.allclose(
        alone.stimulus(X, memory=False)[:, alone.connectome.populations["sensory"]],
        amplitude * row)


def test_the_founder_named_is_the_founder_to_the_bit(tmp_path):
    named, plain = compose("code"), compose()
    assert life(named) == life(plain)
    assert named.describe() == plain.describe()
    a, b = named.save(tmp_path / "a.npz"), plain.save(tmp_path / "b.npz")
    assert a.read_bytes() == b.read_bytes()
    record = saved(a)
    assert "cortex_reads" not in record and "features" not in record
    assert "cortex_reads" not in record["composition"]


def test_describe_names_what_the_cortex_reads():
    assert compose().describe()["layout"]["code"]["cortex"] == "code"
    assert "cortex_reads" not in compose().describe()["genes"]
    for reads in ("both", "observation"):
        described = compose(reads).describe()
        assert described["layout"]["code"]["cortex"] == reads
        assert described["genes"]["cortex_reads"] == reads
        assert described["layout"]["composition"]["cortex_reads"] == reads
    assert cd.Brain.compose(inputs=4, actions=2).describe()["layout"]["code"]["cortex"] == (
        "observation")


# -- save and load


@pytest.mark.parametrize("reads", ["both", "observation"])
def test_a_brain_saves_what_its_cortex_reads_and_continues(tmp_path, reads):
    brain = compose(reads)
    life(brain)
    path = brain.save(tmp_path / "brain.npz")
    record = saved(path)
    assert record["format"] == "cadence-brain/5" and record["features"] == ["cortex_reads"]
    assert record["cortex_reads"] == reads and record["composition"]["cortex_reads"] == reads
    again = cd.Brain.load(path)
    assert again.describe() == brain.describe()
    assert life(again, 4) == life(brain, 4)
    a, b = again.save(tmp_path / "again.npz"), brain.save(tmp_path / "brain2.npz")
    if not kernel._FUSED:  # the NumPy loop continues to the bit
        assert a.read_bytes() == b.read_bytes()
        return
    # the compiled kernel orders its sums by the machine's compiled code: the same acts above,
    # and every array within its last bits (as #346 reads a held message's continuation)
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        assert left.files == right.files
        for name in left.files:
            x, y = left[name], right[name]
            if x.dtype.kind in "fc":
                assert np.allclose(x, y, rtol=1e-12, atol=1e-12), name
            elif name != "brain":  # the record holds diagnostics read from the settles
                assert np.array_equal(x, y), name


def test_a_release_without_the_feature_refuses_the_file_by_its_name(tmp_path, monkeypatch):
    path = compose("both").save(tmp_path / "both.npz")
    monkeypatch.setattr(generic, "FEATURES", generic.FEATURES - {"cortex_reads"})
    with pytest.raises(ValueError, match="'cortex_reads'"):
        cd.Brain.load(path)


def test_a_file_whose_record_disagrees_with_itself_is_refused(tmp_path):
    path = compose("both").save(tmp_path / "both.npz")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for change, match in (
        (lambda r: r.update(cortex_reads="observation"), "cortex_reads"),
        (lambda r: r["composition"].update(cortex_reads="observation"), "cortex_reads"),
        (lambda r: r.update(cortex_reads="senses"), "cortex_reads"),
    ):
        record = json.loads(str(arrays["brain"]))
        change(record)
        target = tmp_path / "changed.npz"
        np.savez_compressed(target, **{**arrays, "brain": np.array(json.dumps(record))})
        with pytest.raises(ValueError, match=match):
            cd.Brain.load(target)


# -- an anatomy of one's own, grow, graft


def test_a_custom_anatomy_with_the_observation_region_reads_both(tmp_path):
    genome = Genome.composed(40, 3, code=64, cortex_reads="both", seed=2)
    assert [r.name for r in genome.regions][-1] == "observation"
    brain = cd.Brain.compose(40, 3, connectome=develop(genome, seed=2), code=64,
                             cortex_reads="both", seed=2, backend="cpu")
    assert brain.describe()["layout"]["code"]["cortex"] == "both"
    again = cd.Brain.load(brain.save(tmp_path / "anatomy.npz"))
    assert again.describe()["layout"]["code"]["cortex"] == "both"
    lacking = develop(Genome.composed(40, 3, code=64, seed=2), seed=2)
    with pytest.raises(ValueError, match="population named 'observation' of 40 neurons"):
        cd.Brain.compose(40, 3, connectome=lacking, code=64, cortex_reads="both", seed=2)


@pytest.mark.parametrize("reads", ["both", "observation"])
def test_grow_keeps_what_the_cortex_reads_and_the_greedy_choices(reads):
    brain = compose(reads)
    life(brain)
    grown = brain.grow(efference_amplitude=1.0)
    assert grown.describe()["layout"]["code"]["cortex"] == reads
    brain.reset()
    grown.reset()
    assert np.array_equal(grown.act(X, greedy=True), brain.act(X, greedy=True))
    with pytest.raises(ValueError, match="cortex_reads changes the brain's own loop"):
        brain.grow(cortex_reads="code")
    with pytest.raises(ValueError, match="cortex_reads is a construction setting"):
        brain.retune(cortex_reads="code")


def test_graft_carries_the_observation_region_between_brains_that_read_both():
    donor, host = compose("both"), compose("both", seed=9)
    life(donor)
    grafted = host.graft(donor, ["observation", "module_0"])
    observed = grafted.connectome.populations["observation"]
    first = grafted.connectome.populations["module_0"]
    pairs = np.isin(grafted.connectome.pre, observed) & np.isin(grafted.connectome.post, first)
    taught = np.isin(donor.connectome.pre, observed) & np.isin(donor.connectome.post, first)
    assert np.array_equal(grafted.learner.brain.efficacy[pairs],
                          donor.learner.brain.efficacy[taught])
    with pytest.raises(ValueError, match="observation"):
        compose().graft(donor, ["observation", "module_0"])


# -- refusals


@pytest.mark.parametrize(("options", "match"), [
    ({"inputs": 40}, "compose the brain with code=units"),
    ({"inputs": 40, "learned_code": 32}, "a learned code is what the cortex reads"),
    ({"inputs": (8, 8), "code": 64}, "an image brain's cortex reads the image"),
    ({"inputs": (16, 16), "learned_fields": 4}, "an image brain's cortex reads the image"),
    ({"inputs": 40, "code": 64, "reads": "senses"}, '"code" \\(the founder\\), "observation"'),
])
def test_cortex_reads_is_refused_where_it_reads_nothing(options, match):
    reads = options.pop("reads", "both")
    with pytest.raises(ValueError, match=match):
        cd.Brain.compose(actions=3, cortex_reads=reads, **options)


def test_a_held_message_hears_the_code_and_is_refused_beside_the_observation():
    both = cd.Brain.compose(inputs=40, actions=3, code=64, cortex_reads="both", message=24,
                            message_boundary=39, message_boundary_holds=True, longterm=8, seed=0,
                            backend="cpu")
    assert both.message.inputs == 64 and len(both.connectome.populations["longterm"]) == 64
    with pytest.raises(ValueError, match='compose the brain with cortex_reads="both"'):
        cd.Brain.compose(inputs=40, actions=3, code=64, cortex_reads="observation", message=24,
                         seed=0, backend="cpu")


def test_unset_is_the_founder():
    assert cd.Brain.compose(inputs=40, actions=3, code=64, cortex_reads=None, seed=0).describe(
    )["layout"]["code"]["cortex"] == "code"


def test_the_constructor_takes_cortex_reads_with_the_inputs_it_needs():
    connectome = develop(Genome.composed(40, 3, code=64, seed=0), seed=0)
    with pytest.raises(ValueError, match="pass no inputs"):
        cd.Brain(connectome, code=64, inputs=40, cortex_reads="observation")
    plain = develop(Genome.composed(40, 3, seed=0), seed=0)
    with pytest.raises(ValueError, match="pass inputs"):
        cd.Brain(plain, code=64, cortex_reads="both")
    reads = cd.Brain(plain, code=64, cortex_reads="observation")  # the sensory region: x
    assert reads.describe()["layout"]["code"]["cortex"] == "observation"


@pytest.mark.parametrize("reads", ["both", "observation"])
def test_lives_refuses_it_by_name(reads):
    with pytest.raises(ValueError, match="cortex_reads"):
        cd.Lives(40, 3, 2, code=64, cortex_reads=reads)


def test_a_trained_module_is_refused_beside_it(tmp_path):
    donor = cd.Brain.compose(inputs=6, actions=3, modules=(8,), seed=0, backend="cpu")
    path = donor.save_module(tmp_path / "m.npz", name="m", inputs=["sensory"],
                             published=["association"])
    with pytest.raises(ValueError, match="without code"):
        cd.Brain.compose(inputs=40, actions=3, code=64, cortex_reads="both",
                         trained=[cd.load_module(path)])


# -- with records of bound moments, a map and the devices


def test_records_of_bound_moments_key_alike_and_complete_with_both():
    things, ear = Things({"hue": (0.0, 1.0), "shape": [(0.0, 1.0)] * 3}), Ear(frames=8, bands=10)
    brains = [cd.Brain.compose(senses={"eyes": things.widths, "ear": ear.width}, actions=2,
                               code=256, bound=64, voice="ear", seed=0, backend="cpu",
                               **({} if reads is None else {"cortex_reads": reads}))
              for reads in (None, "both")]
    cup = things.rows({"hue": [0.85], "shape": [[0.6, 0.3, 0.2]]})
    word = ear.code(np.random.default_rng(0).normal(size=(20, 10)))
    for brain in brains:
        for _ in range(3):
            brain.act({"eyes": cup, "ear": word}, greedy=True)
    code, both = brains
    count = code.bound.store.count
    assert both.bound.store.count == count
    assert np.array_equal(both.bound.store.keys[:count], code.bound.store.keys[:count])
    back = both.complete({"ear": word})["eyes"]
    assert np.allclose(back["shape"], code.complete({"ear": word})["eyes"]["shape"])
    assert float(both.say({"eyes": cup})[0] @ word[0]) > 0.9


def test_a_map_read_joins_the_observation_region_with_the_senses():
    brain = cd.Brain.compose(senses={"touch": 8}, actions=2, code=128, trained=["place"],
                             cortex_reads="both", seed=0, backend="cpu")
    assert len(brain.connectome.populations["observation"]) == brain.inputs == 8 + 848
    brain.space.moment(seen=[(0, [0.3, 0.6])])
    drive = brain.stimulus({"touch": np.eye(8)[[2]]}, memory=False)
    observed = brain.connectome.populations["observation"]
    read = brain.space.observation()
    amplitude = brain.brain.neuron_model.stimulus_amplitude
    assert np.allclose(drive[0, observed][8:], amplitude * read)


@pytest.mark.parametrize("backend", ["mlx", "torch"])
def test_a_device_acts_as_the_cpu_does(backend):
    if backend not in cd.available_backends():
        pytest.skip(f"{backend} is not installed")
    rng = np.random.default_rng(0)
    xs = rng.random((4, 3, 40))
    acts = {}
    for where in ("cpu", backend):
        brain = cd.Brain.compose(inputs=40, actions=3, code=64, cortex_reads="both", seed=0,
                                 backend=where)
        out = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x in xs:
                brain.lesson(x, [0, 1, 2])
                out.append(brain.act(x, greedy=True))
        acts[where] = np.concatenate(out)
    assert np.mean(acts["cpu"] == acts[backend]) >= 0.9


def test_a_copy_reads_as_the_brain_does():
    brain = compose("both")
    life(brain)
    twin = copy.deepcopy(brain)
    assert life(twin, 3) == life(brain, 3)
