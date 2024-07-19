import statistics
import networkx as nx
import numpy as np
import random


class EpistemicAdvantageModel:
    def __init__(self):
        # Define model type
        self.model_type = "variation2"  # can be "base", "variation1", "variation2"
        # Define Parameters
        self.num_agents = (
            40  # number of nodes in network, alternates between 3, 6, 12, 18
        )
        # % population marginalized, alternates between 1/6, 1/3, 1/2, 2/3
        self.proportion_marginalized = 1 / 6
        self.num_pulls = 1  # number of arm pulls, alternates between 1, 5, 10, 20
        # actual probability of B arm yielding success, alternates between .51, .55, .6, .7, .8
        self.objective_b = 0.51
        self.p_ingroup = 0.7
        self.p_outgroup = 0.3
        self.degree_devaluation = 0.2

        # initialize the graph
        self.graph: nx.Graph = None

    def initialize_graph(self):
        if self.model_type == "variation1":
            # Graph for variation1 will not be complete, edges will be added dynamically
            self.graph = nx.Graph()
            self.graph.add_nodes_from((x for x in range(1, self.num_agents)))
        else:
            self.graph = nx.complete_graph(self.num_agents)
        # Initialize all the nodes to this initial data

        for node in self.graph.nodes():
            initial_data = {
                # bandit arm A is set to a 0.5 success rate in the decision process
                "a_success_rate": 0.5,
                # bandit arm B is a learned parameter for the agent. Initialize randomly
                "b_success_rate": random.uniform(0.01, 0.99),
                # agent evidence learned, will be used to update their belief and others in the network
                "b_evidence": None,
                # population type, 'marginalized' or 'dominant'
                "type": "dominant"
                if random.random() > self.proportion_marginalized
                else "marginalized",
            }

            self.graph.nodes[node].update(initial_data)
        if self.model_type == "variation1":
            for potential_neighbor in self.graph.nodes():
                # each node is connected to itself
                if potential_neighbor == node:
                    self.graph.add_edge(node, node)
                # potential connection to an outgroup agent
                elif (
                    self.graph.nodes[potential_neighbor]["type"]
                    != self.graph.nodes[node]["type"]
                ):
                    if random.random() < self.p_outgroup:
                        self.graph.add_edge(node, potential_neighbor)
                # potential connection to an ingroup agent
                else:
                    if random.random() < self.p_ingroup:
                        self.graph.add_edge(node, potential_neighbor)

        return self.graph

    def timestep(self):
        # run the experiments in all the nodes
        for _node, node_data in self.graph.nodes(data=True):
            # agent pulls the "a" bandit arm
            if node_data["a_success_rate"] > node_data["b_success_rate"]:
                # agent won't have any new evidence gathered for b
                node_data["b_evidence"] = None

            # agent pulls the "b" bandit arm
            else:
                # agent collects evidence
                node_data["b_evidence"] = int(
                    np.random.binomial(self.num_pulls, self.objective_b, size=None)
                )

        # define function to calculate posterior belief
        def calculate_posterior(
            prior_belief: float, num_evidence: float, devalue=False
        ) -> float:
            # Calculate likelihood, will be either the success rate
            pEH_likelihood = (self.objective_b**num_evidence) * (
                (1 - self.objective_b) ** (self.num_pulls - num_evidence)
            )

            # Calculate normalization constant
            if devalue:
                pE_evidence = 1 - self.degree_devaluation * (1 - prior_belief)
            else:
                pE_evidence = (pEH_likelihood * prior_belief) + (
                    (1 - self.objective_b) ** num_evidence
                ) * (self.objective_b ** (self.num_pulls - num_evidence)) * (
                    1 - prior_belief
                )

            # Calculate posterior belief using Bayes' theorem
            posterior = (pEH_likelihood * prior_belief) / pE_evidence

            return posterior

        # update the beliefs, based on evidence and neighbors
        for node, node_data in self.graph.nodes(data=True):
            neighbors = self.graph.neighbors(node)
            # update belief of "b" on own evidence gathered
            if node_data["b_evidence"] is not None:
                node_data["b_success_rate"] = calculate_posterior(
                    node_data["b_success_rate"], node_data["b_evidence"]
                )

            # update node belief of "b" based on evidence gathered by neighbors
            for neighbor_node in neighbors:
                neighbor_evidence = self.graph.nodes[neighbor_node]["b_evidence"]
                neighbor_type = self.graph.nodes[neighbor_node]["type"]

                # update from all neighbors if current node is marginalized
                if node_data["type"] == "marginalized" and neighbor_evidence:
                    node_data["b_success_rate"] = calculate_posterior(
                        node_data["b_success_rate"], neighbor_evidence
                    )

                # if current node is dominant and we have variation 2, then devalue marginalized testimony
                else:
                    if self.model_type == "variation2" and neighbor_evidence:
                        node_data["b_success_rate"] = calculate_posterior(
                            node_data["b_success_rate"], neighbor_evidence, devalue=True
                        )
                    # otherwise, if we have the base model, ignore marginalized testimony completely
                    else:
                        if neighbor_type != "marginalized" and neighbor_evidence:
                            node_data["b_success_rate"] = calculate_posterior(
                                node_data["b_success_rate"], neighbor_evidence
                            )

        return self.graph


# Get all keys that have numerical values in the graph
model = EpistemicAdvantageModel()
initial = model.initialize_graph()
median_vals = []
numerical_keys = {
    key
    for node, data in initial.nodes(data=True)
    for key in data
    if isinstance(data[key], (int, float, complex))
}
print("initial graph data", initial.nodes(data=True))
print("/n")
timesteps = 50
# print("b belief at before", [data['b_success_rate'] for node,data in initial.nodes(data=True)])
for _ in range(timesteps):
    updated_graph = model.timestep()
    timestep_medians = {}
    print(
        "b belief at timestep",
        _,
        [data["b_success_rate"] for node, data in updated_graph.nodes(data=True)],
    )
    for key in numerical_keys:
        values = [
            data[key]
            for node, data in updated_graph.nodes(data=True)
            if key in data and data[key] is not None
        ]
        new_median = statistics.median(values) if values else 0
        timestep_medians[key] = new_median
    median_vals.append(timestep_medians)
# print("final graph data", updated_graph.nodes(data=True))
print("median data", [data["b_success_rate"] for data in median_vals])

print("GRAPH_NODES:", model.graph.nodes(data=True, default={}))
print("GRAPH_EDGES:", model.graph.edges(data=True))
