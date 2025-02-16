
import numpy as np
import torch


class TSPDataset(torch.utils.data.Dataset):
  def __init__(self, data_file):
    self.data_file = data_file
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    tour -= 1

    return points, tour

  def __getitem__(self, idx):
    points, tour = self.get_example(idx)
    # Return a densely connected graph
    adj_matrix = np.zeros((points.shape[0], points.shape[0]))
    for i in range(tour.shape[0] - 1):
      adj_matrix[tour[i], tour[i + 1]] = 1
    # return points, adj_matrix, tour
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)),
        torch.from_numpy(points).float(),
        torch.from_numpy(adj_matrix).float(),
        torch.from_numpy(tour).long(),
    )

