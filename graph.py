import glob
import random

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from numpy import ndarray
from pandas import DataFrame
from matplotlib import pyplot as plt
import networkx as nx
from scipy.spatial import distance as distance_calculator
# %load_ext line_profiler

random.seed(42)

features0: DataFrame = pd.concat(
    [pd.read_csv(filename, header=None, dtype=float) for filename in glob.glob("data/0/0/*.csv")])
features0.insert(8, "Label", np.zeros(len(features0)), True)
features1: DataFrame = pd.concat(
    [pd.read_csv(filename, header=None, dtype=float) for filename in glob.glob("data/1/1/*.csv")])
features1.insert(8, "Label", np.ones(len(features1)), True)
features2: DataFrame = pd.concat(
    [pd.read_csv(filename, header=None, dtype=float) for filename in glob.glob("data/2/2/*.csv")])
features2.insert(8, "Label", np.full(len(features2), 2), True)
features3: DataFrame = pd.concat(
    [pd.read_csv(filename, header=None, dtype=float) for filename in glob.glob("data/3/3/*.csv")])
features3.insert(8, "Label", np.full(len(features3), 3), True)
features4: DataFrame = pd.concat(
    [pd.read_csv(filename, header=None, dtype=float) for filename in glob.glob("data/4/4/*.csv")])
features4.insert(8, "Label", np.full(len(features4), 4), True)
features5: DataFrame = pd.concat(
    [pd.read_csv(filename, header=None, dtype=float) for filename in glob.glob("data/bigger/bigger/*.csv")])
features5.insert(8, "Label", np.full(len(features5), 5), True)

all_data = pd.concat([features0, features1, features2, features3, features4, features5])

all_data_grouped = all_data.groupby(6)
print(len(all_data_grouped))

data = [frame.to_numpy() for (_, frame) in all_data_grouped]
first_frame = data[0]
plt.scatter(first_frame[:, 5], first_frame[:, 4])
plt.xlim(-10, 4)
plt.ylim(2, 14)
plt.show()

#TODO: make function that keeps consequent frames together depending of frame_depth

shuffled = np.asarray([frame for (_, frame) in all_data_grouped], dtype="object")
random.shuffle(shuffled)
train, test, val = np.split(shuffled, [int(0.7 * len(shuffled)), int(0.85 * len(shuffled))], axis=0)

# Split the train, test and validation sets to features and labels
X_train = [df.drop([7,'Label'], axis=1) for df in train]
Y_train = [df['Label'] for df in train]

X_test = [df.drop([7,'Label'], axis=1) for df in test]
Y_test = [df['Label'] for df in test]

X_val = [df.drop([7,'Label'], axis=1) for df in val]
Y_val = [df['Label'] for df in val]

class Graph:
    def __init__(self, nodes, edges, adjacency_list, global_attr):
        self.nodes = nodes
        self.edges = edges
        self.adjacency_list = adjacency_list
        self.global_attr = global_attr

    @staticmethod
    def nearest_neighbours(point, previous_frame):
        distances = []
        for p in previous_frame:
            distances.append(np.linalg.norm(point-p))
        return np.asarray(distances)

    @staticmethod
    def connect_frames(current_frame, previous_frame, k: int, start_index: int):
        edges = []
        adjacency_list = []
        previous_nodes = np.arange(len(previous_frame))
        for i, point in enumerate(current_frame):
            # distances = Graph.nearest_neighbours(point, previous_frame)
            distances = distance_calculator.cdist([point], previous_frame, 'euclidean')[0]
            idx = distances.argsort()[::-1]
            distances = distances[idx]
            previous_nodes = previous_nodes[idx]
            for j in range(k):
                edges.append(distances[j])
                adjacency_list.append((i+start_index, previous_nodes[j]+start_index+len(current_frame)))
        return edges, adjacency_list

    @staticmethod
    def create_graph_list(frames:[DataFrame], k = 1, frame_depth = 2):
        graphs = []
        for i, frame in enumerate(frames[frame_depth:]):
            nodes = []
            edges = []
            adjacency_list = []
            relevant_frames = frames[i: i + frame_depth + 1]
            point_data = [rf.drop(6, axis=1) for rf in relevant_frames]
            time_distance = relevant_frames[-1].iloc[0, -1] - relevant_frames[0].iloc[0, -1]
            if time_distance > frame_depth * 3:
                continue
            relevant_frames_array = [df.to_numpy() for df in point_data]
            start_index = 0
            for depth in range(frame_depth):
                pairwise_edges, pairwise_adjacency_list = \
                    Graph.connect_frames(relevant_frames_array[frame_depth-depth], relevant_frames_array[frame_depth-depth-1], k, start_index)
                start_index += len(relevant_frames[frame_depth-depth])
                edges.extend(pairwise_edges)
                adjacency_list.extend(pairwise_adjacency_list)
                nodes.extend(relevant_frames_array[frame_depth-depth])
            nodes.extend(relevant_frames_array[0])
            graphs.append(Graph(nodes, edges, adjacency_list, []))
        return graphs

class GraphVisualization:

    def __init__(self):
        self.visual = []
        self.nodes = []

    def addEdge(self, a, b, d):
        temp = [a, b, {"dist": d}]
        self.visual.append(temp)

    def addNode(self, a):
        self.nodes.append(a)

    def visualize(self):
        gg = nx.Graph()
        gg.add_nodes_from(self.nodes)
        gg.add_edges_from(self.visual)
        nx.draw_networkx(gg)
        plt.show()


# G = GraphVisualization()

sorted_X_train= sorted(X_train,key=lambda x:x[6].max(axis=0))
tiny = [df.iloc[:15, 2:] for df in sorted_X_train[2:10]]
frame_graphs = Graph.create_graph_list(sorted_X_train)
print(len(frame_graphs))
# for i, _ in enumerate(frame_graphs[0].nodes):
#     G.addNode(i)
# for i, adj in enumerate(frame_graphs[0].adjacency_list):
#     G.addEdge(adj[0], adj[1], frame_graphs[0].edges[i])
#
# G.visualize()



tiny_array = [t.to_numpy() for t in tiny]

plt.scatter(tiny_array[0][:, 3], tiny_array[0][:, 2], c='blue')
plt.scatter(tiny_array[1][:, 3], tiny_array[1][:, 2], c='red')
plt.scatter(tiny_array[2][:, 3], tiny_array[2][:, 2], c='green')
# plt.xlim(-10, 4)
# plt.ylim(2, 14)
plt.show()