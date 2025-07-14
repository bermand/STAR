from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle
from .utils import rodrigues, quat_feat, with_zeros
from ..config import cfg


class STAR(nn.Module):
    def __init__(self, gender='female', num_betas=10, use_cuda=True, model_path=None):
        """
        STAR human body model. 
        Args:
            gender (str): 'male', 'female', or 'neutral'
            num_betas (int): number of shape parameters
            use_cuda (bool): whether to use CUDA (GPU). If False, run on CPU. Default: True.
            model_path (str, optional): custom path to the model file. If None, use default for gender.
        """
        super(STAR, self).__init__()

        if gender not in ['male', 'female', 'neutral']:
            raise RuntimeError('Invalid Gender')

        # Device assignment must come before any .to(device) or buffer registration
        device_type = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_type)

        # Model path logic
        if model_path is not None:
            path_model = model_path
        elif gender == 'male':
            path_model = cfg.path_male_star
        elif gender == 'female':
            path_model = cfg.path_female_star
        else:
            path_model = cfg.path_neutral_star

        if not os.path.exists(path_model):
            raise RuntimeError('Path does not exist %s' % (path_model))

        star_model = np.load(path_model, allow_pickle=True)
        J_regressor = star_model['J_regressor']
        self.num_betas = num_betas

        # Model sparse joints regressor, regresses joints location from a mesh
        self.register_buffer('J_regressor', torch.tensor(J_regressor, dtype=torch.float32, device=self.device))

        # Model skinning weights
        self.register_buffer('weights', torch.tensor(star_model['weights'], dtype=torch.float32, device=self.device))
        # Model pose corrective blend shapes
        self.register_buffer('posedirs', torch.tensor(star_model['posedirs'].reshape((-1, 93)), dtype=torch.float32,
                                                      device=self.device))
        # Mean Shape
        self.register_buffer('v_template',
                             torch.tensor(star_model['v_template'], dtype=torch.float32, device=self.device))
        # Shape corrective blend shapes
        self.register_buffer('shapedirs',
                             torch.tensor(np.array(star_model['shapedirs'][:, :, :num_betas]), dtype=torch.float32,
                                          device=self.device))
        # Mesh triangles
        self.register_buffer('faces', torch.from_numpy(star_model['f'].astype(np.int64)))
        self.f = star_model['f']
        # Kinematic tree of the model
        self.register_buffer('kintree_table', torch.from_numpy(star_model['kintree_table'].astype(np.int64)))

        id_to_col = {self.kintree_table[1, i].item(): i for i in range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in range(1, self.kintree_table.shape[1])]))

        self.verts = None
        self.J = None
        self.R = None

    def forward(self, pose, betas, trans):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, self.num_betas)[None, :].expand(batch_size, -1, -1)
        beta = betas[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        J = torch.einsum('bik,ji->bjk', [v_shaped, self.J_regressor])

        pose_quat = quat_feat(pose.view(-1, 3)).view(batch_size, -1)
        pose_feat = torch.cat((pose_quat[:, 4:], beta[:, 1]), 1)

        R = rodrigues(pose.view(-1, 3)).view(batch_size, 24, 3, 3)
        R = R.view(batch_size, 24, 3, 3)

        posedirs = self.posedirs[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, pose_feat[:, :, None]).view(-1, 6890, 3)

        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890, batch_size, 4,
                                                                                             4).transpose(0, 1)
        rest_shape_h = torch.cat([v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        v = v + trans[:, None, :]
        v.f = self.f
        v.v_posed = v_posed
        v.v_shaped = v_shaped

        root_transform = with_zeros(torch.cat((R[:, 0], J[:, 0][:, :, None]), 2))
        results = [root_transform]
        for i in range(0, self.parent.shape[0]):
            transform_i = with_zeros(
                torch.cat((R[:, i + 1], J[:, i + 1][:, :, None] - J[:, self.parent[i]][:, :, None]), 2))
            curr_res = torch.matmul(results[self.parent[i]], transform_i)
            results.append(curr_res)
        results = torch.stack(results, dim=1)
        posed_joints = results[:, :, :3, 3]
        v.J_transformed = posed_joints + trans[:, None, :]
        return v