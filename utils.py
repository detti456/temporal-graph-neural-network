import random
from typing import  List

import numpy as np
import math
import pandas as pd
from pandas import DataFrame
from scipy.spatial import distance as distance_calculator
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import frame_loader as frame_loader

def split_with_chunks(data_array: np.array, labels: List[int], num_chunks:int, val_frac:float, test_frac:float):
    """
    Divides the dataset into train, test and validition sets. 
    This done by dividing each label's data into n chunks 
    and from each chunk the last val_frac and test_frac amount is moved to validation and test sets.

    :param data_array: The array containing the data
    :param labels: The labels found in the dataset
    :param num_chunks: The number of chunks to divide the data to
    :param val_frac: Fraction of validation set size (between 0 and 1)
    :param test_frac: Fraction of test set size (between 0 and 1)
    :return: The train, validation and test set
    """
    train_frac = 1 - val_frac - test_frac
    train = []
    test = []
    val =[]
    label_values = data_array[:,0,5]

    for label in labels:
        itemindex = np.where(label_values == label)
        frames = data_array[itemindex[0]] 
        chunk_size = int(len(frames) / num_chunks)
        chunk_indexes = [i*chunk_size for i in range(num_chunks)]
        chunk_indexes.append(len(frames))

        for chunk in range(num_chunks):
            data_chunk = frames[chunk_indexes[chunk] : chunk_indexes[chunk+1]]
            train.extend(data_chunk[:int(train_frac * len(data_chunk))])
            test.extend(data_chunk[int(train_frac * len(data_chunk)):int((train_frac + test_frac) * len(data_chunk))])
            val.extend(data_chunk[int((train_frac + test_frac) * len(data_chunk)):])

    assert len(train)+len(val)+len(test) == len(data_array)
    return np.array(train), np.array(val), np.array(test)

def split_with_chunks_list(data_array: list, labels: List[int], num_chunks:int, val_frac:float, test_frac:float):
    """
    Divides the dataset into train, test and validition sets. 
    This done by dividing each label's data into n chunks 
    and from each chunk the last val_frac and test_frac amount is moved to validation and test sets.

    :param data_array: The array containing the data
    :param labels: The labels found in the dataset
    :param num_chunks: The number of chunks to divide the data to
    :param val_frac: Fraction of validation set size (between 0 and 1)
    :param test_frac: Fraction of test set size (between 0 and 1)
    :return: The train, validation and test set
    """
    train_frac = 1 - val_frac - test_frac
    train = []
    test = []
    val =[]
    label_values =  np.array([data[0][5] for data in data_array])

    for label in labels:
        itemindex = np.where(label_values == label)
        frames = [data_array[index] for index in itemindex[0]]
        chunk_size = int(len(frames) / num_chunks)
        chunk_indexes = [i*chunk_size for i in range(num_chunks)]
        chunk_indexes.append(len(frames))

        for chunk in range(num_chunks):
            data_chunk = frames[chunk_indexes[chunk] : chunk_indexes[chunk+1]]
            train.extend(data_chunk[:int(train_frac * len(data_chunk))])
            test.extend(data_chunk[int(train_frac * len(data_chunk)):int((train_frac + test_frac) * len(data_chunk))])
            val.extend(data_chunk[int((train_frac + test_frac) * len(data_chunk)):])

    assert len(train)+len(val)+len(test) == len(data_array)
    return train, val, test

def min_max_normalization(frames:DataFrame, selected_cols:List[str]):
    """
    Normalizes the data by appliying min-max normalization on each feature given in 'selected_cols'.

    :param frames: The dataset to normalize
    :param selected_cols: The names of the columns to normalize
    :return: The normalized features together with the sequence numbers and labels.
    """
    scaler = MinMaxScaler().fit(frames[selected_cols])
    normalized_frames = scaler.transform(frames[selected_cols])
    print("Max features: "+str(scaler.data_max_))
    print("Min features: "+str(scaler.data_min_))
    df_normalized = pd.DataFrame(normalized_frames, columns = selected_cols)
    return pd.concat([df_normalized, frames[['current_frame','Label']]], axis=1)

def connect_frames(current_frame:List[np.array], next_frame:List[np.array], k: int, start_index: int, mode:str="desc"):
    """
    Calculates the edges between two frames. 
    
    :param current_frame: The current frame containing parameters to calculate the distance based on 
    :param next_frame: The previous frame containing parameters to calculate the distance based on 
    :param k: The number of neighbours to have in the graph
    :param start_index: The index to start labeling the nodes from
    :param descending: Indicates whether to connect nearest neighbours or furthest neighbours
    :return: The values (euclidean distance between points) of the edges and an adjacency list containing the nodes that are connected in the graph. 
            It also returns a boolean indicating whether the creation was successful or not.
    """
    if mode == "one_to_one":
        current_nodes = np.arange(start_index, start_index+len(current_frame)) 
        next_nodes = np.arange(start_index+len(current_frame), start_index+len(current_frame)+len(next_frame))
        if len(current_nodes) < k or len(next_nodes) < k:
            return [], [], False
        edges = [math.dist(curr,next_frame[min(len(next_frame)-(k-i),max(i,curr_index-(math.ceil(k/2)-1)+i))]) for curr_index, curr in enumerate(current_frame) for i in range(k)]
        adjacency_list = [(curr,next_nodes[min(len(next_frame)-(k-i),max(i,curr_index-(math.ceil(k/2)-1)+i))]) for curr_index, curr in enumerate(current_nodes) for i in range(k)]
        return edges, adjacency_list, True
    edges = []
    adjacency_list = []
    current_nodes = np.arange(start_index, start_index+len(current_frame)) 
    next_nodes = np.arange(start_index+len(current_frame), start_index+len(current_frame)+len(next_frame))
    distances_all  = distance_calculator.cdist(current_frame, next_frame, 'euclidean')
    for i in range(len(current_frame)):
        distances = distances_all[i]
        if len(distances) < k:
            return [], [], False
        if mode == "desc":
            idx = distances.argsort()[::-1]
        elif mode == "asc":
            idx = distances.argsort()
        elif mode == "random":
            idx = np.arange(len(distances))
            k_idx = random.sample(list(idx), k)
            idx[:k] = k_idx
        else:
            raise Exception("No such mode")
        distances = distances[idx]
        ordered_next_nodes = next_nodes[idx]
        
        edges.extend(distances[:k])
        for j in range(k):
            adjacency_list.append((current_nodes[i], ordered_next_nodes[j]))
    return edges, adjacency_list, True

def create_graph_list_with_overlap(frames:np.array, selected_cols:List[str], device:str, size:int, split:int,
                                   k:int = 3, frame_depth:int = 2, mode:str="desc"):
    """
    Creates a list of Data objects that represents the graphs built from the input data. 
    The edges in the graph connects the frames to the previous frame by connecting each 
    points in a frame to it's nearest/furthest neighbour in the previous frame. 
    The nodes contain information about the selected columns. 
    The edges store information about the eucledian distance between the points.
    
    :param frames: Input data grouped and sorted by the frame number
    :param device: The device to store the graphs on (cuda or cpu)
    :param selected_cols: The names of the columns to make nodes out of
    :param k: The number of neighbours to connect each points to
    :param frame_depth: The depth of the graph (number of previos nodes)
    :param descending: Indicates whether to connect nearest neighbours or furthest neighbours
    :return: A list of Data objects, containing information about the created graphs
    """
    parts = int(size/split)
    if len(frames) == 0:
        return []
    graphs = []
    for i, frame in enumerate(frames[frame_depth:]):
        nodes = []
        edges = []
        adjacency_list = []
        relevant_frames = frames[i: i + frame_depth + 1]
        point_data = relevant_frames[:,:,0:4]

        # further split frames into more frames
        point_data_array = np.array([[f[int((p*size)/parts):int(((p+1)*size)/parts)] for p in range(parts)] for f in point_data])
        point_data_array = point_data_array.reshape((frame_depth+1)*parts,-1,len(selected_cols))

        # only make graphs if the gap between any two frames is at most 9 and all frames have the same label
        frame_diff = [relevant_frames[i+1,0, 4] - relevant_frames[i,0, 4] for i in range(frame_depth)]
        if max(frame_diff) > 9 or relevant_frames[-1,0, 5] != relevant_frames[0,0, 5]:
            continue

        start_index = 0
        new_depth = len(point_data_array)-1
        for depth in range(new_depth):
            # calculate the distance for the edges based on the x and y coordinates
            pairwise_edges, pairwise_adjacency_list, success = \
                connect_frames(point_data_array[new_depth-depth][:,2:4], point_data_array[new_depth-depth-1][:,2:4], k, start_index, mode)
            if not success:
                break
            start_index += len(point_data_array[new_depth-depth])
            edges.extend(pairwise_edges)
            adjacency_list.extend(pairwise_adjacency_list)
            nodes.extend(point_data_array[new_depth-depth])
        if not success:
                continue
        nodes.extend(point_data_array[0])
        label = frame[0,5]
        data = Data(x=torch.tensor(np.array(nodes), dtype=torch.float32, device=device),
                    edge_index=torch.tensor(np.array(adjacency_list), dtype=torch.int64, device=device).t().contiguous(),
                    edge_attr=torch.tensor(np.array(edges), dtype=torch.float32, device=device),
                    y=torch.tensor(int(label), dtype=torch.int64, device=device))
        
        graphs.append(data)
    return graphs

def create_graph_list_with_overlap_list(frames:list, selected_cols:List[str], device:str, size:int, split:int,
                                   k:int = 3, frame_depth:int = 2, mode:str="desc"):
    """
    Creates a list of Data objects that represents the graphs built from the input data. 
    The edges in the graph connects the frames to the previous frame by connecting each 
    points in a frame to it's nearest/furthest neighbour in the previous frame. 
    The nodes contain information about the selected columns. 
    The edges store information about the eucledian distance between the points.
    
    :param frames: Input data grouped and sorted by the frame number
    :param device: The device to store the graphs on (cuda or cpu)
    :param selected_cols: The names of the columns to make nodes out of
    :param k: The number of neighbours to connect each points to
    :param frame_depth: The depth of the graph (number of previos nodes)
    :param descending: Indicates whether to connect nearest neighbours or furthest neighbours
    :return: A list of Data objects, containing information about the created graphs
    """
    if len(frames) == 0:
        return []
    graphs = []
    for i, frame in enumerate(frames[frame_depth:]):
        nodes = []
        edges = []
        adjacency_list = []
        relevant_frames = frames[i: i + frame_depth + 1]
        point_data = [[r[0:4] for r in rel] for rel in relevant_frames]

        # only make graphs if the gap between any two frames is at most 9 and all frames have the same label
        frame_diff = [relevant_frames[i+1][0][4] - relevant_frames[i][0][4] for i in range(frame_depth)]
        if max(frame_diff) > 9 or relevant_frames[-1][0][5] != relevant_frames[0][0][5]:
            continue

        # point_data_array = [df.to_numpy() for df in point_data]
        start_index = 0
        new_depth = frame_depth
        for depth in range(new_depth):
            #calculate the distance for the edges based on the x and y coordinates
            pairwise_edges, pairwise_adjacency_list, success = \
                connect_frames(np.array([data[2:4] for data in point_data[new_depth-depth]]), np.array([data[2:4] for data in point_data[new_depth-depth-1]]), k, start_index, mode)
            if not success:
                break
            start_index += len(point_data[new_depth-depth])
            edges.extend(pairwise_edges)
            adjacency_list.extend(pairwise_adjacency_list)
            nodes.extend(point_data[new_depth-depth])
        if not success:
                continue
        nodes.extend(point_data[0])
        label = frame[0,5]
        data = Data(x=torch.tensor(np.array(nodes), dtype=torch.float32, device=device),
                    edge_index=torch.tensor(np.array(adjacency_list), dtype=torch.int64, device=device).t().contiguous(),
                    edge_attr=torch.tensor(np.array(edges), dtype=torch.float32, device=device),
                    y=torch.tensor(int(label), dtype=torch.int64, device=device))
        
        graphs.append(data)
    return graphs


def load_graphs(train:List[DataFrame], val:List[DataFrame], test:List[DataFrame], 
                frame_depths:List[int], ks:List[int], selected_cols:List[str], num_chunks:int, size:int, split:int, device:str, mode:str="desc"):
    """
    Loads the graphs if already saved or creates them if not yet saved.

    :param train: The frames in the train set
    :param val: The frames in the validation set
    :param test: The frames in the test set
    :param frame_depths: The depths of the graph (number of previos nodes)
    :param ks: The number of neighbours to connect each points to
    :param selected_cols: The names of the columns to make nodes out of
    :param device: The device to store the graphs on (cuda or cpu)
    :param descending: Indicates whether to connect nearest neighbours or furthest neighbours
    :return: The graphs created from the train, validation and test sets
    """
    graph_sets = []
    for i, data_set in enumerate([train, val, test]):
        generated_graphs = []
        for f in frame_depths:
            for k in ks:
                try:
                    graphs = torch.load(f"data/frame_graphs_k{k}_frame_depth{f}_type{i}.pt")
                    print("File read")
                except Exception as e:
                    graphs = create_graph_list_with_overlap_list(data_set, selected_cols, device, size=size, split=split, k=k, frame_depth=f, mode=mode)
                    torch.save(graphs, f"data/frame_graphs_k{k}_frame_depth{f}_type{i}.pt")
                
                print(f"Number of graphs generated with k = {k} and frame depth = {f} for type {i}: {len(graphs)}")
                generated_graphs.append(graphs)
        graph_sets.append(generated_graphs)
    
    return graph_sets[0], graph_sets[1], graph_sets[2]