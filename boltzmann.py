from modelpy_abm.main import AgentModel
import random


def generateInitialData(model: AgentModel):
    initial_data = {"wealth": 1}
    return initial_data


def generateTimestepData(model: AgentModel):
    graph = model.get_graph()

    for _node, node_data in graph.nodes(data=True):
        if node_data["wealth"] > 0:
            other_agent = random.choice(list(graph.nodes.keys()))
            if other_agent is not None:
                graph.nodes[other_agent]["wealth"] += 1
                node_data["wealth"] -= 1

    model.set_graph(graph)


model = AgentModel()

model.set_initial_data_function(generateInitialData)
model.set_timestep_function(generateTimestepData)
model.update_parameters({"num_nodes": 3})
model.initialize_graph()

# Run for loop for number of timesteps
timesteps = 100

for _ in range(timesteps):
    model.timestep()

# Print results
print(model.get_graph())
