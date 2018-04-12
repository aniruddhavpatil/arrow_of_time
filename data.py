import torch
import numpy as np
from torch.utils.data import Dataset

class DescriptorsDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        vid_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        descriptor = io.imread(vid_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'descriptor': descriptor, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample