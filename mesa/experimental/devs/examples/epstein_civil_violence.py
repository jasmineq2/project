import enum
import math

import mesa
from mesa import Agent, Model
from mesa.experimental.devs.simulator import ABMSimulator
from mesa.space import SingleGrid
from mesa.time import RandomActivation
from mesa.visualization.solara_viz import SolaraViz, make_text


class EpsteinAgent(Agent):
    def __init__(self, unique_id, model, vision, movement):
        super().__init__(unique_id, model)
        self.vision = vision
        self.movement = movement


class AgentState(enum.IntEnum):
    QUIESCENT = 0
    ARRESTED = 1
    ACTIVE = 2


class Citizen(EpsteinAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        unique_id: unique int
        model :
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        regime_legitimacy: Agent's perception of regime legitimacy, equal
            across agents.  Exogenous.
        risk_aversion: Exogenous, drawn from U(0,1).
        threshold: if (grievance - (risk_aversion * arrest_probability)) >
            threshold, go/remain Active
        vision: number of cells in each direction (N, S, E and W) that agent
            can inspect
        condition: Can be "Quiescent" or "Active;" deterministic function of
            greivance, perceived risk, and
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given
            rebellion
    """

    def __init__(
        self,
        unique_id,
        model,
        vision,
        movement,
        hardship,
        regime_legitimacy,
        risk_aversion,
        threshold,
        arrest_prob_constant,
    ):
        """
        Create a new Citizen.
        Args:
            unique_id: unique int
            model : model instance
            hardship: Agent's 'perceived hardship (i.e., physical or economic
                privation).' Exogenous, drawn from U(0,1).
            regime_legitimacy: Agent's perception of regime legitimacy, equal
                across agents.  Exogenous.
            risk_aversion: Exogenous, drawn from U(0,1).
            threshold: if (grievance - (risk_aversion * arrest_probability)) >
                threshold, go/remain Active
            vision: number of cells in each direction (N, S, E and W) that
                agent can inspect. Exogenous.
        """
        super().__init__(unique_id, model, vision, movement)
        self.hardship = hardship
        self.regime_legitimacy = regime_legitimacy
        self.risk_aversion = risk_aversion
        self.threshold = threshold
        self.condition = AgentState.QUIESCENT
        self.grievance = self.hardship * (1 - self.regime_legitimacy)
        self.arrest_probability = None
        self.arrest_prob_constant = arrest_prob_constant
        self.jail_time_remaining = 0

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.condition == AgentState.ARRESTED:
            self.jail_time_remaining -= 1
            if self.jail_time_remaining <= 0:
                self.release_from_jail()
            return

        self.update_neighbors()
        self.update_estimated_arrest_probability()
        net_risk = self.risk_aversion * self.arrest_probability
        if self.grievance - net_risk > self.threshold:
            self.condition = AgentState.ACTIVE
        else:
            self.condition = AgentState.QUIESCENT
        if self.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.model.grid.move_agent(self, new_pos)

    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)
        self.empty_neighbors = [
            c for c in self.neighborhood if self.model.grid.is_cell_empty(c)
        ]

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = len([c for c in self.neighbors if isinstance(c, Cop)])
        actives_in_vision = 1.0  # citizen counts herself
        for c in self.neighbors:
            if isinstance(c, Citizen) and c.condition == AgentState.ACTIVE:
                actives_in_vision += 1
        self.arrest_probability = 1 - math.exp(
            -1 * self.arrest_prob_constant * (cops_in_vision / actives_in_vision)
        )

    def sent_to_jail(self, value):
        self.model.schedule.remove(self)
        self.condition = AgentState.ARRESTED
        self.jail_time_remaining = jail_time

    def release_from_jail(self):
        self.model.schedule.add(self)
        self.condition = AgentState.QUIESCENT


class Cop(EpsteinAgent):
    """
    A cop for life.  No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Attributes:
        unique_id: unique int
        x, y: Grid coordinates
        vision: number of cells in each direction (N, S, E and W) that cop is
            able to inspect
    """

    def __init__(self, unique_id, model, vision, movement, max_jail_term):
        super().__init__(unique_id, model, vision, movement)
        self.max_jail_term = max_jail_term

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        self.update_neighbors()
        active_neighbors = []
        for agent in self.neighbors:
            if isinstance(agent, Citizen) and agent.condition == AgentState.ACTIVE:
                active_neighbors.append(agent)
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            arrestee.sent_to_jail(self.random.randint(0, self.max_jail_term))
        if self.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.model.grid.move_agent(self, new_pos)

    def update_neighbors(self):
        """
        Look around and see who my neighbors are.
        """
        self.neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=self.vision
        )
        self.neighbors = self.model.grid.get_cell_list_contents(self.neighborhood)
        self.empty_neighbors = [
            c for c in self.neighborhood if self.model.grid.is_cell_empty(c)
        ]


class EpsteinCivilViolence(Model):
    """
    Model 1 from "Modeling civil violence: An agent-based computational
    approach," by Joshua Epstein.
    http://www.pnas.org/content/99/suppl_3/7243.full
    Attributes:
        height: grid height
        width: grid width
        citizen_density: approximate % of cells occupied by citizens.
        cop_density: approximate % of cells occupied by cops.
        citizen_vision: number of cells in each direction (N, S, E and W) that
            citizen can inspect
        cop_vision: number of cells in each direction (N, S, E and W) that cop
            can inspect
        legitimacy:  (L) citizens' perception of regime legitimacy, equal
            across all citizens
        max_jail_term: (J_max)
        active_threshold: if (grievance - (risk_aversion * arrest_probability))
            > threshold, citizen rebels
        arrest_prob_constant: set to ensure agents make plausible arrest
            probability estimates
        movement: binary, whether agents try to move at step end
        max_iters: model may not have a natural stopping point, so we set a
            max.
    """

    def __init__(
        self,
        width=40,
        height=40,
        citizen_density=0.7,
        cop_density=0.074,
        citizen_vision=7,
        cop_vision=7,
        legitimacy=0.8,
        max_jail_term=1000,
        active_threshold=0.1,
        arrest_prob_constant=2.3,
        movement=True,
        max_iters=1000,
        seed=None,
    ):
        super().__init__(seed)
        if cop_density + citizen_density > 1:
            raise ValueError("Cop density + citizen density must be less than 1")

        self.width = width
        self.height = height
        self.citizen_density = citizen_density
        self.cop_density = cop_density

        self.max_iters = max_iters

        self.grid = SingleGrid(self.width, self.height, torus=True)
        self.schedule = RandomActivation(self)

        for _, pos in self.grid.coord_iter():
            if self.random.random() < self.cop_density:
                agent = Cop(
                    self.next_id(),
                    self,
                    cop_vision,
                    movement,
                    max_jail_term,
                )
            elif self.random.random() < (self.cop_density + self.citizen_density):
                agent = Citizen(
                    self.next_id(),
                    self,
                    citizen_vision,
                    movement,
                    hardship=self.random.random(),
                    regime_legitimacy=legitimacy,
                    risk_aversion=self.random.random(),
                    threshold=active_threshold,
                    arrest_prob_constant=arrest_prob_constant,
                )
            else:
                continue
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)

        self.datacollector = mesa.DataCollector(
            {"unhappy": "unhappy", "happy": "happy"}
        )
        self.datacollector.collect(self)

        self.running = True

    @property
    def unhappy(self):
        num_unhappy = 0
        for agent in self.schedule.agents:
            if isinstance(agent, Citizen) and agent.condition == AgentState.ACTIVE:
                num_unhappy += 1
        return num_unhappy

    @property
    def happy(self):
        return len(self.schedule.agents) - self.unhappy

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

        if not self.unhappy:
            self.running = False


def agent_portrayal(agent):
    if isinstance(agent, Citizen):
        if agent.condition == AgentState.QUIESCENT:
            color = "tab:blue"
        elif agent.condition == AgentState.ACTIVE:
            color = "tab:red"
        else:  # ARRESTED
            color = "tab:gray"
    elif isinstance(agent, Cop):
        color = "tab:green"
    else:
        color = "tab:purple"  # Fallback color

    return {
        "color": color,
        "size": 50,
    }


def get_citizen_cop_ratio(model):
    if model.schedule is None:
        return "Citizen/Cop Ratio: N/A"
    citizen_count = sum(isinstance(agent, Citizen) for agent in model.schedule.agents)
    cop_count = sum(isinstance(agent, Cop) for agent in model.schedule.agents)
    ratio = citizen_count / cop_count if cop_count > 0 else float("inf")
    return f"Citizen/Cop Ratio: {ratio:.2f}"


page = SolaraViz(
    model_class=EpsteinCivilViolence,
    model_params={
        "width": 40,
        "height": 40,
        "citizen_density": 0.7,
        "cop_density": 0.074,
        "citizen_vision": 7,
        "cop_vision": 7,
        "legitimacy": 0.8,
        "max_jail_term": 1000,
        "active_threshold": 0.1,
        "arrest_prob_constant": 2.3,
        "movement": True,
        "max_iters": 1000,
    },
    measures=[
        make_text(get_citizen_cop_ratio),
    ],
    name="Epstein Civil Violence Model",
    agent_portrayal=agent_portrayal,
)


@solara.component
def App():
    solara.Title("Epstein Civil Violence Model")
    page.show()


if __name__ == "__main__":
    model = EpsteinCivilViolence(seed=15)
    simulator = ABMSimulator()
    simulator.setup(model)
    simulator.run_for(time_delta=100)
    App()
