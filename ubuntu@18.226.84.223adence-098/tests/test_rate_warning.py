"""Normalized-rate diagnostics preserve valid configurations and existing defaults."""

import inspect
import warnings
from dataclasses import replace
from pathlib import Path

import pytest

import cadence as cd


@pytest.mark.parametrize("config_type", [cd.LearnerConfig])
@pytest.mark.parametrize("rate_name", ["eta", "eta_bias"])
@pytest.mark.parametrize("momentum", [0.0, 0.9])
def test_either_large_normalized_rate_warns_at_the_call_site(
    config_type, rate_name: str, momentum: float,
) -> None:
    rates = {"eta": 0.003, "eta_bias": 0.002, rate_name: 0.0500001}
    with pytest.warns(RuntimeWarning, match=rf"{config_type.__name__}.*eta or eta_bias") as caught:
        line = inspect.currentframe().f_lineno + 1
        config = config_type(normalize=0.99, momentum=momentum, **rates)
    normalized = [w for w in caught if "eta or eta_bias above" in str(w.message)]
    assert len(normalized) == 1
    assert Path(normalized[0].filename).resolve() == Path(__file__).resolve()
    assert normalized[0].lineno == line
    assert config.eta == rates["eta"] and config.eta_bias == rates["eta_bias"]
    # A bias rate above the synapse rate is the issue-126 pathology (issue 143 for the
    # actor); it warns too.
    dominating = [w for w in caught if "bias step dominates" in str(w.message)]
    assert len(dominating) == (1 if rate_name == "eta_bias" else 0)


@pytest.mark.parametrize("config_type", [cd.LearnerConfig])
@pytest.mark.parametrize("rate", [0.0, 0.001, 0.003, 0.05])
def test_normalized_rates_through_threshold_are_silent(config_type, rate: float) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config_type(normalize=0.99, momentum=0.9, eta=rate, eta_bias=rate)


def test_unnormalized_defaults_and_large_rates_are_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cd.LearnerConfig()
        cd.LearnerConfig(normalize=0.0, eta=1.0, eta_bias=1.0)
        # the actor steps on its raw contrast: large rates are its own and silent
        cd.ActorCriticConfig()
        cd.ActorCriticConfig(eta=1.0, eta_bias=1.0, eta_critic=1.0)


@pytest.mark.parametrize("config_type", [cd.LearnerConfig])
def test_invalid_configuration_raises_before_warning(config_type) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="momentum"):
            config_type(normalize=0.99, momentum=1.0, eta=0.2)


@pytest.mark.parametrize("config_type", [cd.LearnerConfig])
def test_replacing_config_to_enable_normalization_warns_without_changing_rates(config_type) -> None:
    original = config_type()
    with pytest.warns(RuntimeWarning, match=config_type.__name__):
        normalized = replace(original, normalize=0.99)
    assert original.normalize == 0.0
    assert normalized.eta == original.eta
    assert normalized.eta_bias == original.eta_bias
    with pytest.warns(RuntimeWarning, match=config_type.__name__):
        restored = config_type(**normalized.to_dict())
    assert restored == normalized
