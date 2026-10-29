# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Tests for sparsity scheduling classes."""

import pytest

from coreai_opt.pruning.config import (
    ConstantSparsitySchedule,
    PolynomialDecaySchedule,
    SparsityScheduleBase,
)


class TestConstantSparsitySchedule:
    """Tests for the constant (step-function) sparsity schedule."""

    @pytest.mark.parametrize(
        "begin_step,step_count,target,expected",
        [
            (0, 100, 0.9, 0.9),
            (10, 9, 0.5, 0.0),
            (10, 10, 0.5, 0.5),
        ],
        ids=["b=0-applied-immediately", "b=10-before-begin", "b=10-at-begin"],
    )
    def test_compute_sparsity(
        self, begin_step: int, step_count: int, target: float, expected: float
    ) -> None:
        """Returns 0 before begin_step and target at/after begin_step."""
        schedule = ConstantSparsitySchedule(begin_step=begin_step)
        assert schedule.compute_sparsity(step_count, target) == expected

    def test_invalid_begin_step(self) -> None:
        """Negative begin_step raises ValueError."""
        with pytest.raises(ValueError):
            ConstantSparsitySchedule(begin_step=-1)


class TestPolynomialDecaySchedule:
    """Tests for the polynomial-decay sparsity schedule."""

    @pytest.mark.parametrize(
        "kwargs,target,expected_at_steps",
        [
            (
                {"begin_step": 0, "total_iters": 10, "power": 1.0, "initial_sparsity": 0.0},
                0.5,
                [(0, 0.0), (5, 0.5 * 5 / 9), (10, 0.5), (100, 0.5)],
            ),
            (
                {"begin_step": 100, "total_iters": 10, "power": 3.0, "initial_sparsity": 0.1},
                0.5,
                [
                    (0, 0.1),
                    (99, 0.1),
                    (100, 0.1),
                    (103, 0.5 + (0.1 - 0.5) * (1 - 3 / 9) ** 3),
                    (107, 0.5 + (0.1 - 0.5) * (1 - 7 / 9) ** 3),
                    (109, 0.5),
                    (110, 0.5),
                    (1000, 0.5),
                ],
            ),
            (
                {"begin_step": 0, "total_iters": 20, "power": 3.0, "initial_sparsity": 0.0},
                0.8,
                [
                    (0, 0.0),
                    (5, 0.8 + (0.0 - 0.8) * (1 - 5 / 19) ** 3),
                    (10, 0.8 + (0.0 - 0.8) * (1 - 10 / 19) ** 3),
                    (15, 0.8 + (0.0 - 0.8) * (1 - 15 / 19) ** 3),
                    (20, 0.8),
                    (1000, 0.8),
                ],
            ),
            (
                {"begin_step": 0, "total_iters": 10, "power": 1.0, "update_frequency": 5},
                0.5,
                [
                    (0, 0.0),  # boundary at offset 0
                    (3, 0.0),  # off-boundary → holds the offset-0 value
                    (5, 0.5),  # boundary at offset 5 (i=1, t=1)
                    (7, 0.5),  # off-boundary → holds the offset-5 value
                    (10, 0.5),  # past schedule
                ],
            ),
            (
                # total_iters not a multiple of update_frequency:
                # n_updates = ceil(10/3) = 4, so t maxes out at 1 (no overshoot).
                {"begin_step": 0, "total_iters": 10, "power": 3.0, "update_frequency": 3},
                0.5,
                [
                    (0, 0.5 + (0.0 - 0.5) * (1 - 0 / 3) ** 3),  # i=0, t=0
                    (3, 0.5 + (0.0 - 0.5) * (1 - 1 / 3) ** 3),  # i=1, t=1/3
                    (6, 0.5 + (0.0 - 0.5) * (1 - 2 / 3) ** 3),  # i=2, t=2/3
                    (9, 0.5),  # i=3, t=1 → target (no overshoot)
                    (10, 0.5),  # past schedule
                    (100, 0.5),
                ],
            ),
        ],
        ids=[
            "linear-no-warmup",
            "cubic-with-warmup",
            "cubic-target=0.8",
            "update_frequency=5",
            "non-divisible-total_iters",
        ],
    )
    def test_compute_sparsity(
        self,
        kwargs: dict,
        target: float,
        expected_at_steps: list[tuple[int, float]],
    ) -> None:
        """Schedule values match the expected polynomial-decay formula."""
        schedule = PolynomialDecaySchedule(**kwargs)
        for step, expected in expected_at_steps:
            actual = schedule.compute_sparsity(step, target)
            assert actual == pytest.approx(expected)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"total_iters": 0},
            {"total_iters": -1},
            {"total_iters": 10, "update_frequency": 0},
            {"total_iters": 10, "power": 0},
            {"total_iters": 10, "power": -2.0},
            {"total_iters": 10, "initial_sparsity": 1.5},
            {"total_iters": 10, "initial_sparsity": -0.1},
            {"total_iters": 10, "begin_step": -1},
            {"total_iters": 10, "unknown_field": 1},
        ],
        ids=[
            "missing-total_iters",
            "total_iters=0",
            "total_iters<0",
            "update_frequency=0",
            "power=0",
            "power<0",
            "initial_sparsity>1",
            "initial_sparsity<0",
            "begin_step<0",
            "extra_field",
        ],
    )
    def test_invalid_kwargs(self, kwargs: dict) -> None:
        """Invalid arguments raise ValueError via pydantic validation."""
        with pytest.raises(ValueError):
            PolynomialDecaySchedule(**kwargs)


class TestSparsityScheduleRegistry:
    """Tests for registry lookup and dict-based construction."""

    @pytest.mark.parametrize(
        "data,expected_type,expected_attrs",
        [
            (
                {"type": "constant", "begin_step": 5},
                ConstantSparsitySchedule,
                {"begin_step": 5},
            ),
            (
                {"type": "polynomial_decay", "total_iters": 100, "power": 2.0},
                PolynomialDecaySchedule,
                {"total_iters": 100, "power": 2.0},
            ),
        ],
        ids=["constant", "polynomial_decay"],
    )
    def test_maybe_build_from_dict(
        self,
        data: dict,
        expected_type: type[SparsityScheduleBase],
        expected_attrs: dict,
    ) -> None:
        """Dict construction via ConfigRegistryMixin.maybe_build_from_dict works."""
        instance = SparsityScheduleBase.maybe_build_from_dict(data)
        assert isinstance(instance, expected_type)
        for attr, value in expected_attrs.items():
            assert getattr(instance, attr) == value

    @pytest.mark.parametrize(
        "schedule",
        [
            ConstantSparsitySchedule(begin_step=3),
            PolynomialDecaySchedule(
                begin_step=10,
                total_iters=50,
                power=2.5,
                initial_sparsity=0.1,
                update_frequency=2,
            ),
        ],
        ids=["constant", "polynomial_decay"],
    )
    def test_model_dump_roundtrip(self, schedule: SparsityScheduleBase) -> None:
        """model_validate(model_dump(x)) reconstructs an equal instance."""
        restored = type(schedule).model_validate(schedule.model_dump())
        assert restored == schedule
