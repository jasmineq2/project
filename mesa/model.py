# -*- coding: utf-8 -*-
"""
The model class for Mesa framework.

Core Objects: Model

"""
import time
import random
import json


class Model:
    """ Base class for models. """

    def __new__(cls, *args, **kwargs):
        """Create a new model object and instantiate its RNG automatically."""

        model = object.__new__(cls)  # This only works in Python 3.3 and above
        model._seed = time.time()
        if "seed" in kwargs and kwargs["seed"] is not None:
            model._seed = kwargs["seed"]
        model.random = random.Random(model._seed)
        return model

    def __init__(self):
        """ Create a new model. Overload this method with the actual code to
        start the model.

        Attributes:
            schedule: schedule object
            running: a bool indicating if the model should continue running

        """

        self.running = True
        self.schedule = None
        self.current_id = 0

    def run_model(self):
        """ Run the model until the end condition is reached. Overload as
        needed.

        """
        while self.running:
            self.step()

    def step(self):
        """ A single step. Fill in here. """
        pass

    def next_id(self):
        """ Return the next unique ID for agents, increment current_id"""
        self.current_id += 1
        return self.current_id

    def reset_randomizer(self, seed=None):
        """Reset the model random number generator.

        Args:
            seed: A new seed for the RNG; if None, reset using the current seed
        """

        if seed is None:
            seed = self._seed
        self.random.seed(seed)
        self._seed = seed

    def as_json(self, include_agents: bool = True) -> str:
        """Convert Model attributes to JSON.

        Args:
            include_agents: Whether to include agents

        Returns:
            string representation of attributes and properties

        Notes:
            If an attribute is not JSON-serializable, it is replaced by its
            string representation.

            The JSON representation also includes attributes of base classes, but
            properties of base classes are currently not supported.
        """

        attributes_str = json.dumps(self.__dict__, default=lambda a: str(a))

        properties = {
            key: getattr(self, key)
            for key, value in type(self).__dict__.items()
            if type(value) == property
        }

        properties_str = json.dumps(properties, default=lambda a: str(a))

        model_json = (
            attributes_str[:-1] + ", " + properties_str[1:]
            if properties
            else attributes_str
        )

        if include_agents and hasattr(self, "schedule"):
            model_json = model_json[
                :-1
            ] + ', "step": {currentStep}, "agents": [{agents}]}}'.format(
                agents=", ".join([agent.as_json() for agent in self.schedule.agents]),
                currentStep=self.schedule.steps,
            )

        return model_json
