from models.models  import CNNMap, CNNDiscriminator
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import cycle
from tqdm import tqdm


# =========================================================
# 2) Solver
# =========================================================
class BarycenterFlowSolver:
    def __init__(self, device, lr_g=2e-4, lr_d=2e-4):
        self.device = device
        self.G_f = CNNMap(in_ch=3, base=64).to(device)  # female -> male
        self.G_g = CNNMap(in_ch=3, base=64).to(device)  # male -> female

        self.D_m = CNNDiscriminator(img_ch=3, base=64).to(device)  # judge male
        self.D_f = CNNDiscriminator(img_ch=3, base=64).to(device)  # judge female

        self.opt_G = optim.Adam(list(self.G_f.parameters()) + list(self.G_g.parameters()),
                                lr=lr_g, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(list(self.D_m.parameters()) + list(self.D_f.parameters()),
                                lr=lr_d, betas=(0.5, 0.999))
        self.mse = nn.MSELoss()

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
        loss_d_m = 0.5 * (self.mse(d_m_real, torch.ones_like(d_m_real)) +
                          self.mse(d_m_pred, -torch.ones_like(d_m_pred)))
    
        d_f_real = self.D_f(x0)
        d_f_pred = self.D_f(x0_pred)
        loss_d_f = 0.5 * (self.mse(d_f_real, torch.ones_like(d_f_real)) +
                          self.mse(d_f_pred, -torch.ones_like(d_f_pred)))
    
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


def get_schedules(curr_epoch):
    ot = 0.5 ** curr_epoch
    cycle = 1.0
    dyn = 1.0
    return ot, cycle, dyn