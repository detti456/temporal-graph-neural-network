import glob

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, KMeans
from collections import defaultdict


def load_original_frames():
    """
    In each files, the frame numbers are shifted to come after 
    the largest frame number in the previous file. This way all frame numbers are unique.
    """

    column_names=['range','azimuth','doppler','snr','x','y','current_frame','seq']
    max_frame = -10

    features1: DataFrame = [(pd.read_csv(filename, names=column_names, header=None, dtype=np.float64), filename[14:19]) for filename in glob.glob("data/1/1/*.csv")]
    result = defaultdict(list)
    for i in range(len(features1)):
        current = features1[i]
        result[current[1]].append(current[0])
    for k, v in result.items():
        result[k] = pd.concat(v)
        shift = max_frame+10
        result[k]["current_frame"] += shift
        max_frame = max(result[k]["current_frame"])
        print(max_frame)
    features1 = pd.concat([v for _, v in result.items()])
    features1.insert(8, "Label", np.zeros(len(features1), dtype=int), True)
    

    features2: DataFrame = [(pd.read_csv(filename, names=column_names, header=None, dtype=np.float64), filename[14:19]) for filename in glob.glob("data/2/2/*.csv")]
    result = defaultdict(list)
    for i in range(len(features2)):
        current = features2[i]
        result[current[1]].append(current[0])
    for k, v in result.items():
        result[k] = pd.concat(v)
        shift = max_frame+10
        result[k]["current_frame"] += shift
        max_frame = max(result[k]["current_frame"])
        print(max_frame)
    features2 = pd.concat([v for _, v in result.items()])
    features2.insert(8, "Label", np.full(len(features2), 1, dtype=int), True)


    features3: DataFrame = [(pd.read_csv(filename, names=column_names, header=None, dtype=np.float64), filename[14:19]) for filename in glob.glob("data/3/3/*.csv")]
    result = defaultdict(list)
    for i in range(len(features3)):
        current = features3[i]
        result[current[1]].append(current[0])
    for k, v in result.items():
        result[k] = pd.concat(v)
        shift = max_frame+10
        result[k]["current_frame"] += shift
        max_frame = max(result[k]["current_frame"])
        print(max_frame)
    features3 = pd.concat([v for _, v in result.items()])
    features3.insert(8, "Label", np.full(len(features3), 2, dtype=int), True)


    features4: DataFrame = [(pd.read_csv(filename, names=column_names, header=None, dtype=np.float64), filename[14:19]) for filename in glob.glob("data/4/*.csv")]
    result = defaultdict(list)
    for i in range(len(features4)):
        current = features4[i]
        result[current[1]].append(current[0])
    for k, v in result.items():
        result[k] = pd.concat(v)
        shift = max_frame+10
        result[k]["current_frame"] += shift
        max_frame = max(result[k]["current_frame"])
        print(max_frame)
    features4 = pd.concat([v for _, v in result.items()])
    features4.insert(8, "Label", np.full(len(features4), 3, dtype=int), True)


    features5: DataFrame = [(pd.read_csv(filename, names=column_names, header=None, dtype=np.float64), filename[14:19]) for filename in glob.glob("data/bigger/bigger/*.csv")]
    result = defaultdict(list)
    for i in range(len(features5)):
        current = features5[i]
        result[current[1]].append(current[0])
    for k, v in result.items():
        result[k] = pd.concat(v)
        shift = max_frame+10
        result[k]["current_frame"] += shift
        max_frame = max(result[k]["current_frame"])
        print(max_frame)
    features5 = pd.concat([v for _, v in result.items()])
    features5.insert(8, "Label", np.full(len(features5), 4, dtype=int), True)


    features_bikes: DataFrame = [(pd.read_csv(filename, names=column_names, header=None, dtype=np.float64), filename[14:19]) for filename in glob.glob("data/bikes/bikes/*.csv")]
    result = defaultdict(list)
    for i in range(len(features_bikes)):
        current = features_bikes[i]
        result[current[1]].append(current[0])
    for k, v in result.items():
        result[k] = pd.concat(v)
        shift = max_frame+10
        result[k]["current_frame"] += shift
        max_frame = max(result[k]["current_frame"])
        print(max_frame)
    features_bikes = pd.concat([v for _, v in result.items()])
    features_bikes.insert(8, "Label", np.full(len(features_bikes), 5, dtype=int), True)


    all_data = pd.concat([features1, features2, features3, features4, features5, features_bikes])
    all_data.drop_duplicates(subset=['range','azimuth','doppler','snr','y','x','current_frame','Label'], inplace=True, ignore_index=True)
    print("Number of data points: "+str(len(all_data)))
    print("Largest frame number: "+str(max_frame))

    # all_data.to_csv("data/all_data.csv", index=False, header=False)
    return all_data


def equal_frame_loader(column_names, size):
    all_data = load_original_frames()
    all_data_grouped = all_data.groupby("current_frame")
    data_array = [frame.to_numpy() for (_, frame) in all_data_grouped]

    equal_frame_data = []
    linkages = ["ward", "complete", "average", "single"]
    for frame in data_array:
        if len(frame) <= 10:
            continue
        elif len(frame) > size:
            kmeans = KMeans(n_clusters=size, random_state=42, n_init="auto").fit(frame)
            equal_frame_data.extend(kmeans.cluster_centers_)
        elif len(frame) < size:
            i = 0
            extended_frame = frame.copy()
            cluster = 4
            while len(extended_frame)< size:
                clustering_labels = AgglomerativeClustering(n_clusters = cluster,linkage=linkages[i%4]).fit_predict(extended_frame)
                i+=1
                for label in range(cluster):
                    itemindex = np.where(clustering_labels == label)
                    clustered_frames = extended_frame[itemindex[0]]
                    centroid = np.mean(clustered_frames, axis=0)
                    extended_frame = np.append(extended_frame,centroid.reshape(1,-1), axis=0)
                    if len(extended_frame) == size:
                        break
                cluster += 1
            equal_frame_data.extend(extended_frame)
        else:
            equal_frame_data.extend(frame)

    equal_frames = pd.DataFrame(equal_frame_data, columns = column_names)
    equal_frames.to_csv(f"data/frames_{size}points_all_varied_fixed.csv", index=False, header=False)

    return equal_frames