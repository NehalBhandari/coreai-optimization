# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


from abc import abstractmethod

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from coreai_opt._utils.registry_utils import ConfigRegistryMixin as _ConfigRegistryMixin


class SparsityScheduleBase(BaseModel, _ConfigRegistryMixin):
    """Abstract base for sparsity schedules used by ``MagnitudePruner``.

    A sparsity schedule defines how the sparsity applied during pruning
    evolves over training steps. Instead of applying the full target sparsity
    immediately, a schedule lets sparsity rise gradually so the model can
    adapt to it during training. Each schedule is a pure function of the
    pruner's step count and the spec's target sparsity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @abstractmethod
    def compute_sparsity(
        self,
        step_count: int,
        target_sparsity: float,
    ) -> float:
        """Return the sparsity that should be applied at *step_count*.

        Args:
            step_count (int): The current step count of the pruner
                (monotonically increasing).
            target_sparsity (float): The final sparsity we want to reach at the
                end of the pruning schedule.

        Returns:
            float: The sparsity level to apply at the current step.
        """


@SparsityScheduleBase.register("constant")
class ConstantSparsitySchedule(SparsityScheduleBase):
    """Step function: zero before ``begin_step``, ``target_sparsity`` at and after.

    Attributes:
        begin_step (int): Step at which to switch from 0 to ``target_sparsity``.
            Default: 0.
    """

    begin_step: NonNegativeInt = 0

    def compute_sparsity(
        self,
        step_count: int,
        target_sparsity: float,
    ) -> float:
        return target_sparsity if step_count >= self.begin_step else 0.0


@SparsityScheduleBase.register("polynomial_decay")
class PolynomialDecaySchedule(SparsityScheduleBase):
    r"""Polynomial schedule from ``initial_sparsity`` to ``target_sparsity``.

    Inspired by PyTorch's ``torch.optim.lr_scheduler.PolynomialLR`` and the paper
    `"To prune or not to prune" <https://arxiv.org/pdf/1710.01878.pdf>`_.

    Behavior by step:

    - ``step < begin_step`` → ``initial_sparsity``
    - ``begin_step <= step < begin_step + total_iters`` → scheduled value
    - ``step >= begin_step + total_iters`` → ``target_sparsity``

    Formula at update index :math:`i \in [0, n\_updates - 1]`:

    .. math::

        t = i / \max(n\_updates - 1, 1)

        sparsity = target + (initial - target) \cdot (1 - t)^{power}

    Attributes:
        begin_step (int): Step at which the schedule starts. Default: 0.
        total_iters (int): Length of the schedule in steps. Must be positive.
        power (float): Polynomial exponent. ``1.0`` is linear; higher values
            keep sparsity low for longer before climbing. Default: 3.0.
        initial_sparsity (float): Sparsity before and at the start of the
            schedule, in ``[0, 1]``. Default: 0.0.
        update_frequency (int): Steps between sparsity updates within the
            schedule. Must be >= 1. Default: 1 (update every step).
    """

    begin_step: int = Field(default=0, ge=0)
    total_iters: PositiveInt
    power: PositiveFloat = 3.0
    initial_sparsity: float = Field(default=0.0, ge=0.0, le=1.0)
    update_frequency: PositiveInt = 1

    def compute_sparsity(
        self,
        step_count: int,
        target_sparsity: float,
    ) -> float:
        if step_count < self.begin_step:
            return self.initial_sparsity
        if step_count >= self.begin_step + self.total_iters:
            return target_sparsity
        offset = step_count - self.begin_step
        n_updates = max((self.total_iters - 1) // self.update_frequency + 1, 1)
        i = offset // self.update_frequency
        t = i / max(n_updates - 1, 1)
        return target_sparsity + (self.initial_sparsity - target_sparsity) * (1 - t) ** self.power
