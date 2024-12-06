"""Collection class for managing and querying groups of cells.

The CellCollection class provides a consistent interface for operating on multiple
cells, supporting:
- Filtering and selecting cells based on conditions
- Random cell and agent selection
- Access to contained agents
- Group operations

This is useful for implementing area effects, zones, or any operation that needs
to work with multiple cells as a unit. The collection handles efficient iteration
and agent access across cells. The class is used throughout the cell space
implementation to represent neighborhoods, selections, and other cell groupings.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from random import Random
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from mesa.experimental.cell_space.cell import Cell
    from mesa.experimental.cell_space.cell_agent import CellAgent

T = TypeVar("T", bound="Cell")


class CellCollection(Generic[T]):
    """An immutable collection of cells.

    Attributes:
        cells (List[Cell]): The list of cells this collection represents
        agents (List[CellAgent]) : List of agents occupying the cells in this collection
        random (Random) : The random number generator

    Notes:
        A `UserWarning` is issued if `random=None`. You can resolve this warning by explicitly
        passing a random number generator. In most cases, this will be the seeded random number
        generator in the model. So, you would do `random=self.random` in a `Model` or `Agent` instance.


    """

    def __init__(
        self,
        cells: Mapping[T, list[CellAgent]] | Iterable[T],
        random: Random | None = None,
    ) -> None:
        """Initialize a CellCollection.

        Args:
            cells: cells to add to the collection
            random: a seeded random number generator.
        """
        if isinstance(cells, dict):
            self._cells = cells
        else:
            self._cells = {cell: cell.agents for cell in cells}

        # Get capacity from first cell if collection is not empty
        self._capacity: int | None = (
            next(iter(self._cells.keys())).capacity if self._cells else None
        )

        if random is None:
            warnings.warn(
                "Random number generator not specified, this can make models non-reproducible. Please pass a random number generator explicitly",
                UserWarning,
                stacklevel=2,
            )
            random = Random()
        self.random = random

    def __iter__(self):  # noqa
        return iter(self._cells)

    def __getitem__(self, key: T) -> Iterable[CellAgent]:  # noqa
        return self._cells[key]

    # @cached_property
    def __len__(self) -> int:  # noqa
        return len(self._cells)

    def __repr__(self):  # noqa
        return f"CellCollection({self._cells})"

    @cached_property
    def cells(self) -> list[T]:  # noqa
        return list(self._cells.keys())

    @property
    def agents(self) -> Iterable[CellAgent]:  # noqa
        return itertools.chain.from_iterable(self._cells.values())

    def select_random_cell(self) -> T:
        """Select a random cell."""
        return self.random.choice(self.cells)

    def select_random_agent(self) -> CellAgent:
        """Select a random agent.

        Returns:
            CellAgent instance


        """
        return self.random.choice(list(self.agents))


    def sample_random_cells(self, k:int=1, counts:list|None=None) -> list[Cell]:
        """Sample k unique cells from the cell collection.

        See random.sample for more information.

        Args:
            k: number of cells to sample
            counts: The number of repeated elements, see random.sample for more details

        Returns:
            a list of sampled cells

        """
        return self.random.sample(self.cells, k, counts=counts)

    def choices_random_cells(self, weights: list|None=None, cum_weights:list|None=None, k:int=1) -> list[Cell]:
        """Return a k sized list of population elements chosen with replacement.

        See random.choices for more information.

        Args:
            weights: weights for each cell
            cum_weights: cumulative weights for each cell
            k: number of cells to sample

        Returns:
            a list of chosen cells

        """
        return self.random.choices(list(self._cells.keys()), weights=weights, cum_weights=cum_weights, k=k)

    def select(
        self,
        filter_func: Callable[[T], bool] | None = None,
        at_most: int | float = float("inf"),
    ):
        """Select cells based on filter function.

        Args:
            filter_func: filter function
            at_most: The maximum amount of cells to select. Defaults to infinity.
              - If an integer, at most the first number of matching cells is selected.
              - If a float between 0 and 1, at most that fraction of original number of cells

        Returns:
            CellCollection

        """
        if filter_func is None and at_most == float("inf"):
            return self

        if at_most <= 1.0 and isinstance(at_most, float):
            at_most = int(len(self) * at_most)  # Note that it rounds down (floor)

        def cell_generator(filter_func, at_most):
            count = 0
            for cell in self:
                if count >= at_most:
                    break
                if not filter_func or filter_func(cell):
                    yield cell
                    count += 1

        return CellCollection(cell_generator(filter_func, at_most), random=self.random)
