"""Trained modules: a module file, a brain composed around a module, and a module grafted."""

from __future__ import annotations

import json

import numpy as np
import pytest

import cadence as cd
from cadence import Brain, modules
from cadence.space import Interface, Space
from cadence.space.place import INPUTS, OUTPUTS

PLANE = (-1.0, 1.0, -0.2, 1.4)
SMALL = dict(place=8, moved=5, motion=9, counted=3, unseen=4, heard=(6, 4), felt=4, hidden=16,
             shift=12, spread=6, found=8, keep=4)
PUBLISHED = ("place", *OUTPUTS)


@pytest.fixture(scope="module")
def taught() -> Space:
    """A small place brain after a few lessons, so its synapses are no founder's."""
    space = Space(Interface(plane=PLANE, **SMALL), seed=0, backend="cpu")
    rng = np.random.default_rng(0)
    start = rng.uniform([-0.5, 0.2], [0.5, 1.0], (4, 2))
    space.begin(start, own=[True, False, False, False])
    for t in range(1, 30):
        reports = start + 0.01 * t + rng.normal(0.0, 0.01, (4, 2))
        space.advance(np.ones(4, dtype=bool), reports, rng.uniform(-0.05, 0.05, 2), learn=True)
    return space


@pytest.fixture(scope="module")
def written(taught: Space, tmp_path_factory: pytest.TempPathFactory):
    path = tmp_path_factory.mktemp("module") / "place.npz"
    taught.brain.save_module(
        path, name="place", inputs=INPUTS, published=PUBLISHED,
        regions=(*INPUTS, "hidden", *OUTPUTS), interface=taught.interface.to_dict(),
        card={"trained on": "thirty steps of a test"})
    return path


def test_a_module_file_holds_arrays_and_one_record(taught: Space, written) -> None:
    module = cd.load_module(written)
    interface = taught.interface
    assert module.name == "place" and module.regions == interface.regions
    assert module.inputs == list(INPUTS) and module.published == list(PUBLISHED)
    assert module.width == interface.inputs
    assert Interface.from_dict(module.interface) == interface
    assert module.card == {"trained on": "thirty steps of a test"}
    assert module.record["format"] == "cadence-module/1"
    assert module.record["library"] == cd.__version__
    with np.load(written, allow_pickle=False) as held:  # no pickled object in the file
        assert set(held.files) == {"module", *module.arrays}
        assert all(held[name].dtype != object for name in held.files)
    assert module.arrays["efficacy"].dtype == np.float32
    donor = module.brain()
    assert module.brain() is not donor  # a brain of its own each call
    assert np.allclose(donor.brain.efficacy, taught.brain.brain.efficacy, atol=1e-6)
    assert donor.connectome.populations.keys() == taught.brain.connectome.populations.keys()


def test_a_malformed_module_file_is_refused_by_what_is_wrong(written, tmp_path) -> None:
    with np.load(written, allow_pickle=False) as held:
        arrays = {name: held[name] for name in held.files}
    record = json.loads(str(arrays["module"]))

    def file(changed: dict, **replaced) -> str:
        path = tmp_path / "odd.npz"
        np.savez(path, **{**arrays, "module": np.array(json.dumps({**record, **changed})),
                          **replaced})
        return str(path)

    with pytest.raises(ValueError, match="holds no 'module' record"):
        path = tmp_path / "bare.npz"
        np.savez(path, **{k: v for k, v in arrays.items() if k != "module"})
        cd.load_module(path)
    with pytest.raises(ValueError, match="states format 'cadence-module/1'"):
        cd.load_module(file({"format": "cadence-module/9"}))
    with pytest.raises(ValueError, match="'efficacy' differs from its checksum"):
        cd.load_module(file({}, efficacy=arrays["efficacy"] + 1.0))
    with pytest.raises(ValueError, match="names 'elsewhere'"):
        cd.load_module(file({"published": ["elsewhere"]}))
    with pytest.raises(ValueError, match="no region 'hidden' of 99 neurons"):
        cd.load_module(file({"regions": {**record["regions"], "hidden": 99}}))
    with pytest.raises(ValueError, match="holds no 'inputs'"):
        cd.load_module(file({"inputs": None}))
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, module=np.array({"format": "x"}, dtype=object))
    with pytest.raises(ValueError, match="is no module file"):
        cd.load_module(pickled)
    with pytest.raises(ValueError, match="ships no module 'nothing'"):
        modules.load("nothing")


def test_a_brain_composed_around_a_module_holds_its_regions_and_its_skill(
        taught: Space, written) -> None:
    module = cd.load_module(written)
    host = Brain.compose(inputs=5, actions=3, seed=1, trained=[module], backend="cpu")
    interface = taught.interface
    assert host.describe()["layout"]["inputs"] == 5 + module.width
    populations = host.connectome.populations
    for region, size in module.regions.items():
        assert len(populations[f"place/{region}"]) == size
    columns = module.columns(host)
    assert columns.tolist() == list(range(5, 5 + module.width))
    assert host.describe()["memories"].keys() == Brain.compose(
        inputs=5, actions=3, seed=1, backend="cpu").describe()["memories"].keys()
    # the module's synapses are the donor's and are held; the brain's own are free
    inside = np.isin(host.connectome.pre, populations["place"]) & np.isin(
        host.connectome.post, populations["place"])
    assert inside.sum() == module.brain().connectome.synapses
    assert not host.learner.plastic_synapses[inside].any()
    assert host.learner.plastic_synapses[~inside].any()
    # in the host that took no lesson the regions answer as in their donor
    inside_host = Space(interface, brain=host, module="place")
    in_donor = Space(interface, brain=module.brain())
    rng = np.random.default_rng(3)
    start = rng.uniform([-0.5, 0.2], [0.5, 1.0], (3, 2))
    inside_host.begin(start, own=[True, False, False])
    in_donor.begin(start, own=[True, False, False])
    for t in range(1, 25):
        seen = rng.random(3) < 0.5
        reports = start + 0.01 * t + rng.normal(0.0, 0.01, (3, 2))
        command = rng.uniform(-0.05, 0.05, 2)
        theirs = inside_host.predict(command, beside=rng.random((3, 5)))
        mine = in_donor.predict(command, learn=True)  # the donor settles every row too
        assert np.allclose(theirs, mine, atol=1e-9)
        inside_host.observe(seen, reports)
        in_donor.observe(seen, reports)
    assert inside_host.acted is not None and inside_host.acted.shape == (3,)
    with pytest.raises(ValueError, match="takes no lessons"):
        inside_host.predict(np.zeros(2), learn=True)
    # a lesson of the host's own task leaves the module's synapses as they are
    before = host.brain.efficacy.copy()
    x = np.hstack([rng.random((3, 5)), inside_host.sense(np.zeros(2))])
    for _ in range(5):
        host.lesson(x, [0, 1, 2])
    moved = host.brain.efficacy != before
    assert moved.any() and not moved[inside].any()


def test_a_module_is_grafted_into_a_brain_that_holds_its_regions(taught: Space, written) -> None:
    from cadence.space import anatomy

    module = cd.load_module(written)
    fresh = Brain.compose(taught.interface.inputs, taught.interface.outputs,
                          connectome=anatomy(taught.interface, seed=5),
                          slots=taught.interface.slots, seed=5, episodic=False,
                          teaching="founder", temperature=0.05, backend="cpu")
    grafted = fresh.graft(module)
    assert np.allclose(grafted.brain.efficacy, module.brain().brain.efficacy, atol=1e-6)
    assert not grafted.learner.plastic_synapses.any()  # a module is held unless asked
    free = fresh.graft(module, freeze=False)
    assert free.learner.plastic_synapses.any()
    with pytest.raises(ValueError, match="has no region 'place' for the module 'place'"):
        Brain.compose(inputs=5, actions=3, seed=1, backend="cpu").graft(module)
    narrow = Interface(plane=PLANE, **{**SMALL, "hidden": 12})
    with pytest.raises(ValueError, match="'hidden' holds 12 neurons and the module's 'hidden' 16"):
        Brain.compose(narrow.inputs, narrow.outputs, connectome=anatomy(narrow, seed=5),
                      slots=narrow.slots, seed=5, episodic=False, backend="cpu").graft(module)


def test_a_space_is_saved_as_a_module_and_made_of_one(taught: Space, tmp_path) -> None:
    path = taught.save(tmp_path / "place.npz", card={"founder": 0})
    module = cd.load_module(path)
    assert module.name == "place" and module.published == ["place", *OUTPUTS]
    assert np.isclose(module.interface["noise"], np.sqrt(taught.noise))
    alone = Space.load(path)
    assert alone.interface == taught.interface and np.isclose(alone.noise, taught.noise)
    host = Brain.compose(inputs=4, actions=2, seed=3, trained=[module], backend="cpu")
    hosted = Space.load(module, brain=host)
    assert hosted.module == "place" and np.isclose(hosted.noise, taught.noise)
    start = np.array([[0.2, 0.6]])
    alone.begin(start, own=True)
    hosted.begin(start, own=True)
    a = alone.advance([False], np.zeros((1, 2)), np.array([0.04, -0.03]))
    b = hosted.advance([False], np.zeros((1, 2)), np.array([0.04, -0.03]))
    assert np.allclose(a, b, atol=1e-9) and not np.allclose(a, start)


def test_a_module_of_another_versions_regions_is_refused_with_its_reason(taught: Space,
                                                                         tmp_path) -> None:
    old = taught.brain.save_module(
        tmp_path / "old.npz", name="place", inputs=INPUTS, published=("place", "shift"),
        regions=(*INPUTS, "hidden", "shift", "spread", "found"),  # a file from before ``keep``
        interface={"interface": taught.interface.to_dict(), "noise": 0.01})
    with pytest.raises(ValueError, match="was made with cadence .* another version's"):
        Space.load(old)


def test_a_tracked_space_keeps_the_synapses_a_lesson_moved() -> None:
    space = Space(Interface(plane=PLANE, **SMALL), seed=0, backend="cpu", track=True)
    rng = np.random.default_rng(1)
    start = np.array([[0.0, 0.5], [0.3, 0.9]])
    space.begin(start)
    for t in range(1, 12):
        space.advance([True, True], start + 0.01 * t + rng.normal(0.0, 0.01, (2, 2)), learn=True)
    assert len(space.changed) > 0 and len(space.changed) == len(space.changes)
    assert space.changed.max() < space.brain.connectome.synapses and (space.changes != 0).all()
    outputs = set(space.brain.connectome.populations["motor"])
    assert set(space.brain.connectome.post[space.changed].tolist()) <= outputs


def test_trained_is_refused_where_it_does_not_fit(written) -> None:
    module = cd.load_module(written)
    with pytest.raises(ValueError, match="vector brain"):
        Brain.compose(inputs=(6, 6), actions=3, trained=[module])
    with pytest.raises(ValueError, match="without code"):
        Brain.compose(inputs=5, actions=3, code=32, trained=[module])
    with pytest.raises(ValueError, match="list of trained modules"):
        Brain.compose(inputs=5, actions=3, trained=["place"])
    with pytest.raises(ValueError, match="share one name"):
        Brain.compose(inputs=5, actions=3, trained=[module, module])


def test_a_brain_without_modules_is_composed_as_before() -> None:
    plain = Brain.compose(inputs=5, actions=3, seed=2, backend="cpu")
    named = Brain.compose(inputs=5, actions=3, seed=2, backend="cpu", trained=None)
    empty = Brain.compose(inputs=5, actions=3, seed=2, backend="cpu", trained=[])
    assert plain.describe() == named.describe() == empty.describe()
    assert np.array_equal(plain.brain.efficacy, named.brain.efficacy)
    assert np.array_equal(plain.brain.efficacy, empty.brain.efficacy)


@pytest.fixture
def index_type_int32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(np, "intp", np.int32)


def test_a_module_loads_and_a_host_is_composed_under_a_32_bit_index_type(
        index_type_int32: None, taught: Space, written) -> None:
    module = cd.load_module(written)
    host = Brain.compose(inputs=5, actions=3, seed=1, trained=[module], backend="cpu")
    space = Space(taught.interface, brain=host, module="place")
    space.begin(np.array([[0.1, 0.5]]), own=True)
    place = space.advance([False], np.zeros((1, 2)), np.array([0.03, -0.02]),
                          cues=np.array([[-2.0, 1.0]]))
    assert place.shape == (1, 2) and np.isfinite(place).all()
    assert module.columns(host).dtype == np.int32
