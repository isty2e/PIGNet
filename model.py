from argparse import Namespace
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedGAT(nn.Module):
    def __init__(self, n_in_feature: int, n_out_feature: int):
        super().__init__()

        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_in_feature + n_out_feature, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        h = self.W(x)
        e = torch.einsum("ijl,ikl->ijk", (torch.matmul(h, self.A), h))
        e = e + e.permute((0, 2, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 1e-6, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = attention * adj
        h_prime = F.relu(torch.einsum("aij,ajk->aik", (attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x, h_prime], -1))).repeat(
            1, 1, x.size(-1)
        )
        new_x = coeff * x + (1 - coeff) * h_prime
        return new_x


class InteractionNet(nn.Module):
    def __init__(self, n_atom_feature: int):
        super().__init__()

        self.W = nn.Linear(n_atom_feature, n_atom_feature)
        self.M = nn.Linear(n_atom_feature, n_atom_feature)
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)

    def forward(self, x1: Tensor, x2: Tensor, valid_edge: Tensor) -> Tensor:
        new_edge = x2.unsqueeze(1).repeat(1, x1.size(1), 1, 1)

        m1 = self.W(x1)
        m2 = (self.M(new_edge) * valid_edge.unsqueeze(-1)).max(2)[0]
        x_cat = F.relu(m1 + m2)
        feature_size = x_cat.size(-1)
        x_cat = self.C(x_cat.reshape(-1, feature_size), x1.reshape(-1, feature_size))
        x_cat = x_cat.reshape(x1.size(0), x1.size(1), x1.size(2))
        return x_cat


class PIGNet(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args
        self.node_embedding = nn.Linear(54, args.dim_gnn, bias=False)

        self.gconv = nn.ModuleList(
            [GatedGAT(args.dim_gnn, args.dim_gnn) for _ in range(args.n_gnn)]
        )
        if args.interaction_net:
            self.interaction_net = nn.ModuleList(
                [InteractionNet(args.dim_gnn) for _ in range(args.n_gnn)]
            )

        self.cal_vdw_interaction_A = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.cal_vdw_interaction_B = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Tanh(),
        )
        self.cal_vdw_interaction_N = nn.Sequential(
            nn.Linear(args.dim_gnn * 2, args.dim_gnn),
            nn.ReLU(),
            nn.Linear(args.dim_gnn, 1),
            nn.Sigmoid(),
        )
        self.hbond_coeff = nn.Parameter(torch.tensor([1.0]))
        self.hydrophobic_coeff = nn.Parameter(torch.tensor([0.5]))
        self.vdw_coeff = nn.Parameter(torch.tensor([1.0]))
        self.torsion_coeff = nn.Parameter(torch.tensor([1.0]))
        self.rotor_coeff = nn.Parameter(torch.tensor([0.5]))

    def cal_hbond(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = dm * A / -0.7
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hbond_coeff * self.hbond_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_hydrophobic(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        A: Tensor,
    ) -> Tensor:
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )
        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm = dm - dm_0

        retval = (-dm + 1.5) * A
        retval = retval.clamp(min=0.0, max=1.0)
        retval = retval * -(self.hydrophobic_coeff * self.hydrophobic_coeff)
        retval = retval.sum(-1).sum(-1).unsqueeze(-1)
        return retval

    def cal_vdw_interaction(
        self,
        dm: Tensor,
        h: Tensor,
        ligand_vdw_radii: Tensor,
        target_vdw_radii: Tensor,
        ligand_valid: Tensor,
        target_valid: Tensor,
    ) -> Tensor:
        ligand_valid_ = ligand_valid.unsqueeze(2).repeat(1, 1, target_valid.size(1))
        target_valid_ = target_valid.unsqueeze(1).repeat(1, ligand_valid.size(1), 1)
        ligand_vdw_radii_ = ligand_vdw_radii.unsqueeze(2).repeat(
            1, 1, target_vdw_radii.size(1)
        )
        target_vdw_radii_ = target_vdw_radii.unsqueeze(1).repeat(
            1, ligand_vdw_radii.size(1), 1
        )

        B = self.cal_vdw_interaction_B(h).squeeze(-1) * self.args.dev_vdw_radius
        dm_0 = ligand_vdw_radii_ + target_vdw_radii_ + B
        dm_0[dm_0 < 0.0001] = 1
        N = self.args.vdw_N
        vdw_term1 = torch.pow(dm_0 / dm, 2 * N)
        vdw_term2 = -2 * torch.pow(dm_0 / dm, N)

        A = self.cal_vdw_interaction_A(h).squeeze(-1)
        A = A * (self.args.max_vdw_interaction - self.args.min_vdw_interaction)
        A = A + self.args.min_vdw_interaction

        energy = vdw_term1 + vdw_term2
        energy = energy.clamp(max=100)
        energy = energy * ligand_valid_ * target_valid_
        energy = A * energy
        energy = energy.sum(1).sum(1).unsqueeze(-1)
        return energy

    def cal_distance_matrix(
        self, ligand_pos: Tensor, target_pos: Tensor, dm_min: float
    ) -> Tensor:
        p1_repeat = ligand_pos.unsqueeze(2).repeat(1, 1, target_pos.size(1), 1)
        p2_repeat = target_pos.unsqueeze(1).repeat(1, ligand_pos.size(1), 1, 1)
        dm = torch.sqrt(torch.pow(p1_repeat - p2_repeat, 2).sum(-1) + 1e-10)
        replace_vec = torch.ones_like(dm) * 1e10
        dm = torch.where(dm < dm_min, replace_vec, dm)
        return dm

    def forward(
        self, sample: Dict[str, Any], DM_min: float = 0.5, cal_der_loss: bool = False
    ) -> Tuple[Tensor]:
        (
            ligand_h,
            ligand_adj,
            target_h,
            target_adj,
            interaction_indice,
            ligand_pos,
            target_pos,
            rotor,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_valid,
            target_valid,
            ligand_non_metal,
            target_non_metal,
            _,
            _,
        ) = sample.values()

        # feature embedding
        ligand_h = self.node_embedding(ligand_h)
        target_h = self.node_embedding(target_h)

        # distance matrix
        ligand_pos.requires_grad = True
        dm = self.cal_distance_matrix(ligand_pos, target_pos, DM_min)

        # GatedGAT propagation
        for idx in range(len(self.gconv)):
            ligand_h = self.gconv[idx](ligand_h, ligand_adj)
            target_h = self.gconv[idx](target_h, target_adj)
            ligand_h = F.dropout(
                ligand_h, training=self.training, p=self.args.dropout_rate
            )
            target_h = F.dropout(
                target_h, training=self.training, p=self.args.dropout_rate
            )

        # InteractionNet propagation
        if self.args.interaction_net:
            adj12 = dm.clone().detach()

            adj12[adj12 > 5] = 0
            adj12[adj12 > 1e-3] = 1
            adj12[adj12 < 1e-3] = 0

            for idx in range(len(self.interaction_net)):
                new_ligand_h = self.interaction_net[idx](
                    ligand_h,
                    target_h,
                    adj12,
                )
                new_target_h = self.interaction_net[idx](
                    target_h,
                    ligand_h,
                    adj12.permute(0, 2, 1),
                )
                ligand_h, target_h = new_ligand_h, new_target_h
                ligand_h = F.dropout(
                    ligand_h, training=self.training, p=self.args.dropout_rate
                )
                target_h = F.dropout(
                    target_h, training=self.training, p=self.args.dropout_rate
                )

        # concat features
        h1_ = ligand_h.unsqueeze(2).repeat(1, 1, target_h.size(1), 1)
        h2_ = target_h.unsqueeze(1).repeat(1, ligand_h.size(1), 1, 1)
        h_cat = torch.cat([h1_, h2_], -1)

        # compute energy component
        energies = []

        # vdw interaction
        vdw_energy = self.cal_vdw_interaction(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            ligand_non_metal,
            target_non_metal,
        )
        energies.append(vdw_energy)

        # hbond interaction
        hbond = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 0],
        )
        energies.append(hbond)

        # metal interaction
        metal = self.cal_hbond(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 1],
        )
        energies.append(metal)

        # hydrophobic interaction
        hydrophobic = self.cal_hydrophobic(
            dm,
            h_cat,
            ligand_vdw_radii,
            target_vdw_radii,
            interaction_indice[:, 2],
        )
        energies.append(hydrophobic)

        energies = torch.cat(energies, -1)
        # rotor penalty
        if not self.args.no_rotor_penalty:
            energies = energies / (
                1 + self.rotor_coeff * self.rotor_coeff * rotor.unsqueeze(-1)
            )

        # derivatives
        if cal_der_loss:
            gradient = torch.autograd.grad(
                energies.sum(), ligand_pos, retain_graph=True, create_graph=True
            )[0]
            der1 = torch.pow(gradient.sum(1), 2).mean()
            der2 = torch.autograd.grad(
                gradient.sum(), ligand_pos, retain_graph=True, create_graph=True
            )[0]
            der2 = -der2.sum(1).sum(1).mean()
        else:
            der1 = torch.zeros_like(energies).sum()
            der2 = torch.zeros_like(energies).sum()

        return energies, der1, der2
