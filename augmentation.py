
import math
import numpy as np
import torch

def gaussian_noise(batch_data,mean = 0, std_deviation=0.01):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    gaussian_noise = np.random.normal(mean, std_deviation, (B,N, 2))
    
    batch_data[:,:,2:4] += gaussian_noise
    return batch_data

    
class NoiseAddition(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, data):
        all_features = data[:,:,0:4]
        modified = gaussian_noise(all_features)
        return np.append(modified,data[:,:,4:], axis=2)
    

    
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx2 array, original batch of point clouds
        Return:
          BxNx2 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(len(batch_data)):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval],
                                    [sinval, cosval]])
        rotated_data[k] = np.dot(batch_data[k], rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C).astype(np.float32), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 2))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data

def gaussian_noise_edge(batch_data,mean = 0, std_deviation=0.01):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    N,E= batch_data.shape
    gaussian_noise = np.random.normal(mean, std_deviation, (N,E))
    
    batch_data += gaussian_noise
    return batch_data

class AugmentationTransformer(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, data):
        all_features = np.array([d.x.cpu().detach().numpy() for d in data])
        edge_attr = np.array([d.edge_attr.cpu().detach().numpy() for d in data])
        pos = all_features[:,:,2:4]

        modified = jitter_point_cloud(all_features)
        noise_edge = gaussian_noise_edge(edge_attr)

        new_data = []
        for i,d in enumerate(data):
            new_d = d.clone()
            new_d.x = torch.from_numpy(modified[i]).to(new_d.x.device)
            new_d.edge_attr = torch.from_numpy(noise_edge[i]).to(new_d.edge_attr.device)
            new_data.append(new_d)
        return new_data
    
def jitter_point_cloud_list(batch_data, mean = 0, std_deviation=0.1):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    for i, f in enumerate(batch_data):
        for j, p in enumerate(f):
            batch_data[i][j] += np.random.normal(mean, std_deviation, len(batch_data[i][j])) 
    return batch_data

def gaussian_noise_edge_list(batch_data,mean = 0, std_deviation=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    for i, _ in enumerate(batch_data):
        batch_data[i] += np.random.normal(mean, std_deviation, len(batch_data[i]))
    
    return batch_data

def restore_edge(nodes, edges):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    dist = []
    for i,_ in enumerate(nodes):
        dist.append(np.empty(len(edges[i][0]), dtype=np.float32))
        for j, (p, q) in enumerate(zip(edges[i][0],edges[i][1])):
            dist[i][j] = math.dist(nodes[i][p], nodes[i][q])
    
    return dist
    
class AugmentationTransformerList(object):
    def __init__(self):
        pass

    def __call__(self, data):
        all_features = [d.x.cpu().detach().numpy() for d in data]
        edge_attr = [d.edge_attr.cpu().detach().numpy() for d in data]

        modified = jitter_point_cloud_list(all_features)
        noise_edge = gaussian_noise_edge_list(edge_attr)

        new_data = []
        for i,d in enumerate(data):
            new_d = d.clone()
            new_d.x = torch.from_numpy(modified[i]).to(new_d.x.device)
            new_d.edge_attr = torch.from_numpy(noise_edge[i]).to(new_d.edge_attr.device)
            new_data.append(new_d)
        return new_data

