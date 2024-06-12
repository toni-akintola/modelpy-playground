from openai import OpenAI
import os
import openai
import requests
from dotenv import load_dotenv

load_dotenv()
# Set your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)
openai.api_key = api_key

client = OpenAI(api_key=api_key)
# # Define the file path
# file_path = 'path/to/your/file.pdf'

# # Upload the PDF file
# with open(file_path, 'rb') as file:
#     response = requests.post(
#         'https://api.openai.com/v1/files',
#         headers={
#             'Authorization': f'Bearer {api_key}',
#         },
#         files={
#             'file': file,
#             'purpose': (None, 'answers'),
#         }
#     )

# file_id = response.json()['id']
# print(f"File uploaded with ID: {file_id}")
code_block = '''
import networkx as nx

# Define class
class MyModel():
    def __init__(self):
        # Define Parameters
        self.num_nodes = 3
        self.graph_type = 'complete' # complete, wheel, or cycle
        
        # NOTE: This graph variable will not be loaded into the 
        # modelpy interface since it is not a string or number
        self.graph: nx.Graph = None

    def initialize_graph(self):
        # initialize graph shape
        if self.graph_type == 'complete':
            self.graph = nx.complete_graph(self.num_nodes)
        elif self.graph_type == 'cycle':
            self.graph = nx.cycle_graph(self.num_nodes)
        else:
            self.graph = nx.wheel_graph(self.num_nodes)
        
        # Initialize sample data for all nodes
        for node in self.graph.nodes():
            initial_data = {
                'data_value': 0,
            }
            self.graph.nodes[node].update(initial_data)

    def timestep(self):
        for _node, node_data in self.graph.nodes(data=True):
            # example mutate the node data
            node_data['data_value'] += node_data['data_value'] + 1
'''
model_overview = '''
There are 6 poker players at a poker table. They each have varying levels of comfort with/aversion to risk. Build a graph-based ABM that can model this environment. Make the number of poker players modular.
'''
prompt = f"Using the following overview of an agent-based model and some reference code as context, build an agent-based model in Python. Here is the code: {code_block}\n Here is the overview of the model: {model_overview} "
# Use the file in a context window
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)

print(response)
print(response.choices[0].message)
