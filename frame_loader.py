import glob

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.cluster import AgglomerativeClustering, KMeans


def load_original_frames():
    """
    In each files, the frame numbers are shifted to come after 
    the largest frame number in the previous file. This way all frame numbers are unique.
    """

    column_names=['range','azimuth','doppler','snr','x','y','current_frame','seq']

    features1: DataFrame = pd.concat(
        [pd.read_csv(filename, names=column_names, header=None, dtype=np.float64) for filename in glob.glob("data/1/1/*.csv")])
    features1.insert(8, "Label", np.zeros(len(features1), dtype=int), True)
    max_frame = max(features1["current_frame"])

    features2: DataFrame = pd.concat(
        [pd.read_csv(filename, names=column_names, header=None, dtype=np.float64) for filename in glob.glob("data/2/2/*.csv")])
    features2.insert(8, "Label", np.full(len(features2), 1, dtype=int), True)
    shift = max_frame+10
    features2["current_frame"] += shift
    max_frame = max(features2["current_frame"])

    features3: DataFrame = pd.concat(
        [pd.read_csv(filename, names=column_names, header=None, dtype=np.float64) for filename in glob.glob("data/3/3/*.csv")])
    features3.insert(8, "Label", np.full(len(features3), 2, dtype=int), True)
    shift = max_frame+10
    features3["current_frame"] += shift
    max_frame = max(features3["current_frame"])

    features4: DataFrame = pd.concat(
        [pd.read_csv(filename, names=column_names, header=None, dtype=np.float64) for filename in glob.glob("data/4/*.csv")])
    features4.insert(8, "Label", np.full(len(features4), 3, dtype=int), True)
    shift = max_frame+10
    features4["current_frame"] += shift
    max_frame = max(features4["current_frame"])

    features5: DataFrame = pd.concat(
        [pd.read_csv(filename, names=column_names, header=None, dtype=np.float64) for filename in glob.glob("data/bigger/bigger/*.csv")])
    features5.insert(8, "Label", np.full(len(features5), 4, dtype=int), True)
    shift = max_frame+10
    features5["current_frame"] += shift
    max_frame = max(features5["current_frame"])

    features_bikes: DataFrame = pd.concat(
        [pd.read_csv(filename, names=column_names, header=None, dtype=np.float32) for filename in glob.glob("data/bikes/bikes/*.csv")])
    features_bikes.insert(8, "Label", np.full(len(features_bikes), 5, dtype=int), True)
    shift = max_frame+10
    features_bikes["current_frame"] += shift
    max_frame = max(features_bikes["current_frame"])

    all_data = pd.concat([features1, features2, features3, features4, features5, features_bikes])
    all_data.drop_duplicates(subset=['range','azimuth','doppler','snr','y','x','current_frame','Label'], inplace=True, ignore_index=True)
    print("Number of data points: "+str(len(all_data)))
    print("Largest frame number: "+str(max_frame))

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
    equal_frames.to_csv(f"data/frames_{size}points_all_varied.csv", index=False, header=False)

    return equal_frames