from mesa import Agent


class SchellingAgent(Agent):
    """Schelling segregation agent."""

    def __init__(self, model, agent_type: int) -> None:
        """Create a new Schelling agent.

        Args:
            model: The model instance the agent belongs to
            agent_type: Indicator for the agent's type (minority=1, majority=0)
        """
        super().__init__(model)
        self.type = agent_type

    def step(self) -> None:
        """Determine if agent is happy and move if necessary."""
        neighbors = self.model.grid.iter_neighbors(
            self.pos, moore=True, radius=self.model.radius
        )

        # Count similar neighbors
        similarBooleanList = [neighbor.type == self.type for neighbor in neighbors]
        similar = sum(similarBooleanList)

        # Count total neighbors
        total_neighbors = len(similarBooleanList)

        # If unhappy, move to a random empty cell:
        if (
            total_neighbors != 0
            and similar / total_neighbors < self.model.homophily_ratio
        ):
            self.model.grid.move_to_empty(self)
        else:
            self.model.happy += 1
