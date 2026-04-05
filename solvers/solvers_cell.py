from models.icnns import ICNN
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle
from tqdm import tqdm


# =========================================================
# 2) Solver
# =========================================================
class Discriminator(nn.Module):
    def __init__(self, input_dim=48, hidden_units=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, hidden_units),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_units, 1)
        )

    def forward(self, x):
        return self.net(x)

class BarycenterFlowSolver:
    def __init__(self, device, config, lr_g=1e-4, lr_d=1e-4):

        self.device = device
        self.G_f = ICNN(input_dim=48, 
                        hidden_units=config.model.hidden_units).to(device)  # source -> target
        self.G_g = ICNN(input_dim=48, 
                        hidden_units=config.model.hidden_units).to(device)  # target -> source

        self.D_m = Discriminator(input_dim=48, hidden_units=64).to(device)  # judge target
        self.D_f = Discriminator(input_dim=48, hidden_units=64).to(device)  # judge source

        self.opt_G = optim.Adam(list(self.G_f.parameters()) + list(self.G_g.parameters()),
                                lr=lr_g, betas=(0.9, 0.95))
        self.opt_D = optim.Adam(list(self.D_m.parameters()) + list(self.D_f.parameters()),
                                lr=lr_d, betas=(0.9, 0.95))
        self.mse = nn.MSELoss()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def train_step(self, x0, x1, lambdas):
        l_ot, l_cycle, l_dyn = lambdas
    
        # -------------------------
        # 1) Update D (marginal)
        # -------------------------
        self.opt_D.zero_grad()
        with torch.no_grad():
            x1_pred = self.G_f(x0)   
            x0_pred = self.G_g(x1)   
    
        d_m_real = self.D_m(x1)
        d_m_pred = self.D_m(x1_pred)
        # loss_d_m = 0.5 * (self.mse(d_m_real, torch.ones_like(d_m_real)) +
        #                   self.mse(d_m_pred, -torch.ones_like(d_m_pred)))
        loss_d_m = 0.5 * (
            self.bce(d_m_real, torch.ones_like(d_m_real)) +
            self.bce(d_m_pred, torch.zeros_like(d_m_pred))
        )
    
        d_f_real = self.D_f(x0)
        d_f_pred = self.D_f(x0_pred)
        # loss_d_f = 0.5 * (self.mse(d_f_real, torch.ones_like(d_f_real)) +
        #                   self.mse(d_f_pred, -torch.ones_like(d_f_pred)))
        loss_d_f = 0.5 * (
            self.bce(d_f_real, torch.ones_like(d_f_real)) +
            self.bce(d_f_pred, torch.zeros_like(d_f_pred))
        )
    
        loss_d = loss_d_m + loss_d_f
        loss_d.backward()
        self.opt_D.step()
    
        # -------------------------
        # 2) Update G
        # -------------------------
        self.opt_G.zero_grad()
        x1_pred = self.G_f(x0)
        x0_pred = self.G_g(x1)
    
        # marginal GAN (generator wants "real" label)
        loss_g_adv = 0.5 * (self.mse(self.D_m(x1_pred), torch.ones_like(self.D_m(x1_pred))) +
                            self.mse(self.D_f(x0_pred), torch.ones_like(self.D_f(x0_pred))))
    
        # cycle
        loss_cycle = torch.mean((x0 - self.G_g(x1_pred))**2) + \
                     torch.mean((x1 - self.G_f(x0_pred))**2)
    
        # -------------------------
        # 3) OT loss (your original one, unchanged)
        # -------------------------
        if l_ot > 0:
            loss_ot = torch.mean((x1_pred - x0)**2) + torch.mean((x0_pred - x1)**2)
        else:
            loss_ot = torch.tensor(0.0, device=self.device)
    
        # total
        loss_total = (l_dyn * loss_g_adv) + (l_cycle * loss_cycle) + (l_ot * loss_ot)
    
        loss_total.backward()
        self.opt_G.step()
    
        return loss_total.item(), loss_d.item(), loss_ot.item()
    
    def evaluate_step(self, x0, x1):
        with torch.no_grad():
            x1_pred = self.G_f(x0)
            x0_pred = self.G_g(x1)
            mmd = self.gaussian_mmd(x1_pred.detach(), x1.detach())
    
        ot_loss = torch.mean((x1_pred - x0)**2) + torch.mean((x0_pred - x1)**2)
        cycle_loss = torch.mean((x0 - self.G_g(x1_pred))**2) + torch.mean((x1 - self.G_f(x0_pred))**2)

        return ot_loss.item(), cycle_loss.item(), mmd.item()

    def gaussian_mmd(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """A compact biased MMD estimate with an RBF kernel."""
        with torch.no_grad():
            z = torch.cat([x, y], dim=0)
            d2 = torch.cdist(z, z, p=2.0).pow(2)
            med = torch.median(d2[d2 > 0]) if (d2 > 0).any() else torch.tensor(1.0, device=z.device)
            sigma2 = med.clamp_min(1e-6)

        def kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.exp(-torch.cdist(a, b, p=2.0).pow(2) / (2.0 * sigma2))

        k_xx = kernel(x, x).mean()
        k_yy = kernel(y, y).mean()
        k_xy = kernel(x, y).mean()
        return k_xx + k_yy - 2.0 * k_xy



def get_schedules(curr_epoch):
    ot = 0.5 ** curr_epoch
    cycle = 1.0
    dyn = 1.0
    return ot, cycle, dyn