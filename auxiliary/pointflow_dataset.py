import os
import sys
import torch
# import pymesh
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
from tqdm import tqdm as tqdm
from utils import AverageValueMeter

# taken from https://github.com/optas/latent_3d_points/blob/8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py
synsetid_to_cate = {
    '02691156': 'airplane',  '02773838': 'bag',        '02801938': 'basket',
    '02808440': 'bathtub',   '02818832': 'bed',        '02828884': 'bench',
    '02834778': 'bicycle',   '02843684': 'birdhouse',  '02871439': 'bookshelf',
    '02876657': 'bottle',    '02880940': 'bowl',       '02924116': 'bus',
    '02933112': 'cabinet',   '02747177': 'can',        '02942699': 'camera',
    '02954340': 'cap',       '02958343': 'car',        '03001627': 'chair',
    '03046257': 'clock',     '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table',     '04401088': 'telephone',  '02946921': 'tin_can',
    '04460130': 'tower',     '04468005': 'train',      '03085013': 'keyboard',
    '03261776': 'earphone',  '03325088': 'faucet',     '03337140': 'file',
    '03467517': 'guitar',    '03513137': 'helmet',     '03593526': 'jar',
    '03624134': 'knife',     '03636649': 'lamp',       '03642806': 'laptop',
    '03691459': 'speaker',   '03710193': 'mailbox',    '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano',     '03938244': 'pillow',     '03948459': 'pistol',
    '03991062': 'pot',       '04004475': 'printer',    '04074963': 'remote_control',
    '04090263': 'rifle',     '04099429': 'rocket',     '04225987': 'skateboard',
    '04256520': 'sofa',      '04330267': 'stove',      '04530566': 'vessel',
    '04554684': 'washer',    '02858304': 'boat',       '02992529': 'cellphone',
}
cate_to_synsetid = {v:k for k,v in synsetid_to_cate.items()}

class Uniform15KPC(Dataset):

    def __init__(self, root_dir, subdirs, tr_sample_size=10000,
            te_sample_size=10000, split='train', scale=1., mesh_ext='.obj',
            normalize_per_shape=False, random_rotation=False, random_subsample=False):
        self.root_dir = root_dir
        self.datapath = root_dir
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.subdirs = subdirs
        self.scale = scale
        self.mesh_ext = mesh_ext
        self.random_rotation = random_rotation
        self.random_subsample = random_subsample

        self.mesh_files = []
        # self.train_points = []
        # self.test_points  = []
        self.all_points = []
        self.all_cates = []
        for subd in tqdm(self.subdirs, desc='Subdirectories', leave=True):
            sub_path = os.path.join(root_dir, subd, self.split)
            cate = synsetid_to_cate[subd]
            for x in tqdm(os.listdir(sub_path), desc='shapes', leave=False):
                if not x.endswith('.npy'):
                    continue

                obj_fname = os.path.join(sub_path, x)
                try:
                    point_cloud = np.load(obj_fname) # (15k, 3)
                except:
                    continue

                # mesh_fname = obj_fname[:-len('.npy')] + mesh_ext
                #
                # if not os.path.isfile(mesh_fname):
                #     print("Not a file:%s %s"%(obj_fname, mesh_fname))
                #     import pdb; pdb.set_trace()
                #     continue

                assert point_cloud.shape[0] == 15000
                self.all_points.append(point_cloud[np.newaxis,...])
                # self.mesh_files.append(mesh_fname)
                self.all_cates.append(cate)

        # Per shape normalization
        self.all_points = np.concatenate(self.all_points) # (N, 15000, 3)
        if normalize_per_shape:
            self.all_points, self.all_points_mean, self.all_points_std = \
                    self._normalize_pc_(self.all_points)
        else:
            self.all_points_mean = np.zeros((self.all_points.shape[0], 1, 3))
            self.all_points_std  = np.ones((self.all_points.shape[0], 1, 1))
        self.train_points = self.all_points[:,:10000]
        self.test_points  = self.all_points[:,10000:]

        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)
        print("Total number of data:%d"%len(self.train_points))
        print("Min number of points: (train)%d (test)%d"\
              %(self.tr_sample_size, self.te_sample_size))
        assert self.scale == 1, "Scale (!= 1) is deprecated"

    def _normalize_pc_(self, pc):
        """ [pc] (B, N, 3) numpy array """
        B, N = pc.shape[:2]
        pc_mean = pc.mean(axis=1).reshape(B, 1, 3)
        pc_std  = pc.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
        pc_out = (pc - pc_mean) / pc_std
        return pc_out, pc_mean, pc_std

    def _unnormalize_pc_(self, pc, mean, std):
        return pc * std + mean

    def get_mesh(self, idx):
        try:
            return pymesh.load_mesh(self.mesh_files[idx])
        except:
            return None

    def get_pc_stats(self, idx):
        m = self.all_points_mean[idx].reshape(1,1,3)
        s = self.all_points_std[idx].reshape(1,1,1)
        return m, s

    def get_mesh_fname(self, idx):
        return self.mesh_files[idx]

    def __len__(self):
        return len(self.train_points)

    def __getitem__(self, idx):
        tr_out = self.train_points[idx]
        if self.random_subsample:
            tr_idxs = np.random.choice(tr_out.shape[0], self.tr_sample_size)
        else:
            tr_idxs = np.arange(self.tr_sample_size)
        tr_out = torch.from_numpy(tr_out[tr_idxs,:]).float()

        te_out = self.test_points[idx]
        if self.random_subsample:
            te_idxs = np.random.choice(te_out.shape[0], self.te_sample_size)
        else:
            te_idxs = np.arange(self.te_sample_size)
        te_out = torch.from_numpy(te_out[te_idxs,:]).float()

        pcl = torch.cat([tr_out, te_out], dim=0)
        cat = self.all_cates[idx]
        return idx, pcl, cat, 0, 0

class ModelNet40PointClouds(Uniform15KPC):

    def __init__(self, root_dir="data/ModelNet40.PC15k",
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 random_subsample=False):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test']
        self.sample_size = tr_sample_size
        self.cates = []
        for cate in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, cate)) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'train')) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'test')):
                self.cates.append(cate)
        assert len(self.cates) == 40, self.cates

        self.gravity_axis = 0
        self.display_axis_order = [0,1,2]
        super(ModelNet40PointClouds, self).__init__(
                root_dir, self.cates, tr_sample_size=tr_sample_size,
                te_sample_size=te_sample_size, split=split, scale=scale,
                normalize_per_shape=normalize_per_shape, mesh_ext='.off',
                random_subsample=random_subsample)


class ModelNet10PointClouds(Uniform15KPC):

    def __init__(self, root_dir="data/ModelNet10.PC15k",
                 tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 random_subsample=False):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test']
        self.cates = []
        for cate in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, cate)) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'train')) \
                    and os.path.isdir(os.path.join(root_dir, cate, 'test')):
                self.cates.append(cate)
        assert len(self.cates) == 10

        self.gravity_axis = 0
        self.display_axis_order = [0,1,2]

        super(ModelNet10PointClouds, self).__init__(
                root_dir, self.cates, tr_sample_size=tr_sample_size,
                te_sample_size=te_sample_size, split=split, scale=scale,
                normalize_per_shape=normalize_per_shape, mesh_ext='.off',
                random_subsample=random_subsample)


class ShapeNet15kPointClouds(Uniform15KPC):

    def __init__(self, root_dir="data/ShapeNetCore.v2.PC15k",
                 categories=['airplane'], tr_sample_size=10000, te_sample_size=2048,
                 split='train', scale=1., normalize_per_shape=False,
                 random_subsample=False):
        self.root_dir = root_dir
        self.split = split
        assert self.split in ['train', 'test', 'val']
        self.tr_sample_size = tr_sample_size
        self.te_sample_size = te_sample_size
        self.cates = categories
        self.cat = self.cates
        self.synset_ids = [cate_to_synsetid[c] for c in self.cates]

        self.perCatValueMeter = {}
        for item in self.cat:
            self.perCatValueMeter[item] = AverageValueMeter()
        self.perCatValueMeter_metro = {}
        for item in self.cat:
            self.perCatValueMeter_metro[item] = AverageValueMeter()

        self.gravity_axis = 1
        self.display_axis_order = [0,2,1]

        super(ShapeNet15kPointClouds, self).__init__(
                root_dir, self.synset_ids,
                tr_sample_size=tr_sample_size,
                te_sample_size=te_sample_size,
                split=split, scale=scale,
                normalize_per_shape=normalize_per_shape,
                random_subsample=random_subsample)

if __name__ == "__main__":
    shape_ds = ShapeNet15kPointClouds(categories=['airplane'], split='val')
    x_tr, x_te = next(iter(shape_ds))
    print(x_tr.shape)
    print(x_te.shape)

