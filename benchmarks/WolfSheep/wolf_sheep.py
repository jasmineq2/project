"""
Wolf-Sheep Predation Model
================================

Replication of the model found in NetLogo:
    Wilensky, U. (1997). NetLogo Wolf Sheep Predation model.
    http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.
"""

import math

from mesa import Model
from mesa.experimental.cell_space import CellAgent, OrthogonalVonNeumannGrid
from mesa.experimental.devs import ABMSimulator


class Animal(CellAgent):
    def __init__(self, unique_id, model, energy, p_reproduce, energy_from_food):
        super().__init__(unique_id, model)
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food

    def random_move(self):
        self.move_to(self.cell.neighborhood().select_random_cell())

    def spawn_offspring(self):
        self.energy /= 2
        offspring = self.__class__(
            self.model.next_id(),
            self.model,
            self.energy,
            self.p_reproduce,
            self.energy_from_food,
        )
        offspring.move_to(self.cell)

    def feed(self):
        ...

    def die(self):
        self.cell.remove_agent(self)
        self.remove()

    def step(self):
        self.random_move()
        self.energy -= 1

        self.feed()

        if self.energy < 0:
            self.die()
        elif self.random.random() < self.p_reproduce:
            self.spawn_offspring()


class Sheep(Animal):
    """
    A sheep that walks around, reproduces (asexually) and gets eaten.

    The init is the same as the RandomWalker.
    """

    def feed(self):
        # If there is grass available, eat it
        grass_patch = next(
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if grass_patch.fully_grown:
            self.energy += self.energy_from_food
            grass_patch.fully_grown = False


class Wolf(Animal):
    """
    A wolf that walks around, reproduces (asexually) and eats sheep.
    """

    def feed(self):
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if len(sheep) > 0:
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food

            # Kill the sheep
            sheep_to_eat.die()


class GrassPatch(CellAgent):
    """
    A patch of grass that grows at a fixed rate and it is eaten by sheep
    """

    @property
    def fully_grown(self):
        return self._fully_grown

    @fully_grown.setter
    def fully_grown(self, value: bool) -> None:
        self._fully_grown = value

        if not value:
            self.model.simulator.schedule_event_relative(
                self.set_fully_grown, self.grass_regrowth_time
            )

    def __init__(self, unique_id, model, fully_grown, countdown, grass_regrowth_time):
        """
        Creates a new patch of grass

        Args:
            grown: (boolean) Whether the patch of grass is fully grown or not
            countdown: Time for the patch of grass to be fully grown again
        """
        super().__init__(unique_id, model)
        self._fully_grown = fully_grown
        self.grass_regrowth_time = grass_regrowth_time

        if not self.fully_grown:
            self.model.simulator.schedule_event_relative(
                self.set_fully_grown, countdown
            )

    def set_fully_grown(self):
        self.fully_grown = True


class WolfSheep(Model):
    """
    Wolf-Sheep Predation Model

    A model for simulating wolf and sheep (predator-prey) ecosystem modelling.
    """

    def __init__(
        self,
        simulator,
        height,
        width,
        initial_sheep,
        initial_wolves,
        sheep_reproduce,
        wolf_reproduce,
        grass_regrowth_time,
        wolf_gain_from_food=13,
        sheep_gain_from_food=5,
        seed=None,
    ):
        """
        Create a new Wolf-Sheep model with the given parameters.

        Args:
            simulator: ABMSimulator instance
            initial_sheep: Number of sheep to start with
            initial_wolves: Number of wolves to start with
            sheep_reproduce: Probability of each sheep reproducing each step
            wolf_reproduce: Probability of each wolf reproducing each step
            wolf_gain_from_food: Energy a wolf gains from eating a sheep
            grass: Whether to have the sheep eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                 once it is eaten
            sheep_gain_from_food: Energy sheep gain from grass, if enabled.
            seed : the random seed
        """
        super().__init__(seed=seed)
        # Set parameters
        self.height = height
        self.width = width
        self.simulator = simulator

        self.initial_sheep = initial_sheep
        self.initial_wolves = initial_wolves
        self.grass_regrowth_time = grass_regrowth_time

        self.sheep_reproduce = sheep_reproduce
        self.wolf_reproduce = wolf_reproduce
        self.grass_regrowth_time = grass_regrowth_time
        self.wolf_gain_from_food = wolf_gain_from_food
        self.sheep_gain_from_food = sheep_gain_from_food

        self.grid = None

    def setup(self):
        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=False,
            capacity=math.inf,
            random=self.random,
        )

        # Create sheep:
        for _ in range(self.initial_sheep):
            pos = (
                self.random.randrange(self.width),
                self.random.randrange(self.height),
            )
            energy = self.random.randrange(2 * self.sheep_gain_from_food)
            sheep = Sheep(
                self.next_id(),
                self,
                energy,
                self.sheep_reproduce,
                self.sheep_gain_from_food,
            )
            sheep.move_to(self.grid[pos])

        # Create wolves
        for _ in range(self.initial_wolves):
            pos = (
                self.random.randrange(self.width),
                self.random.randrange(self.height),
            )
            energy = self.random.randrange(2 * self.wolf_gain_from_food)
            wolf = Wolf(
                self.next_id(),
                self,
                energy,
                self.wolf_reproduce,
                self.wolf_gain_from_food,
            )
            wolf.move_to(self.grid[pos])

        # Create grass patches
        possibly_fully_grown = [True, False]
        for cell in self.grid:
            fully_grown = self.random.choice(possibly_fully_grown)
            if fully_grown:
                countdown = self.grass_regrowth_time
            else:
                countdown = self.random.randrange(self.grass_regrowth_time)
            patch = GrassPatch(
                self.next_id(), self, fully_grown, countdown, self.grass_regrowth_time
            )
            patch.move_to(cell)

    def step(self):
        self.get_agents_of_type(Sheep).shuffle(inplace=True).do("step")
        self.get_agents_of_type(Wolf).shuffle(inplace=True).do("step")


if __name__ == "__main__":
    import time

    simulator = ABMSimulator()
    model = WolfSheep(
        simulator,
        25,
        25,
        60,
        40,
        0.2,
        0.1,
        20,
        seed=15,
    )

    simulator.setup(model)

    start_time = time.perf_counter()
    simulator.run(100)
    print("Time:", time.perf_counter() - start_time)
