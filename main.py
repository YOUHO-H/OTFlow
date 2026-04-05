import torch
from datasets import make_celeba_gender_loaders
from datasets import visualize_cycle
from solvers.solvers import BarycenterFlowSolver, get_schedules
from itertools import cycle
from tqdm import tqdm
from data.cell import load_cell_data

# =========================================================
# 4) Run Experiment
# =========================================================
def run_celeba_female_to_male_experiment():
    print("\n=== Running Experiment: CelebA (x0=female, x1=male) ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 20 
    IMG_SIZE = 64

    female_loader, male_loader = make_celeba_gender_loaders(
        root="./datasets", batch_size=BATCH_SIZE, num_workers=4, img_size=IMG_SIZE, split="train"
    )

    solver = BarycenterFlowSolver(device)
    total_iters = min(len(female_loader), len(male_loader)) * EPOCHS
    epoch_pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

    iter_count = 0
    for epoch in epoch_pbar:
        g_loss_acc, d_loss_acc, ot_loss_acc= 0, 0, 0

        for (x0_f, _), (x1_m, _) in zip(cycle(female_loader), male_loader):
            x0_clean = x0_f.to(device)  # female clean
            x1_clean = x1_m.to(device)  # male clean

            lambdas = get_schedules(epoch)
            weights = lambdas[0:3]

            lg, ld, lot= solver.train_step(x0_clean, x1_clean, weights)

            g_loss_acc += lg
            d_loss_acc += ld
            ot_loss_acc += lot
            iter_count += 1

        steps = len(male_loader)
        epoch_pbar.set_postfix({
            'G': f"{g_loss_acc/steps:.3f}",
            'D': f"{d_loss_acc/steps:.3f}",
            'OT': f"{ot_loss_acc/steps:.3f}",
        })

        if (epoch + 1) % 5 == 0:
            solver.G_f.eval()
            solver.G_g.eval()
            # -------------------------
            # 2) cycle maps
            # -------------------------
            visualize_cycle(
                solver.G_f, solver.G_g,
                x0_clean.to(device),
                n_show=10,
                title=f"Cycle: female x0 -> G_f(x0) -> G_g(G_f(x0))",
                row1_label="female $x_{0}$",
                row2_label="male? $G_f(x_{0})$",
                row3_label="female rec $G_g(G_f(x_{0}))$"
            )
            
            visualize_cycle(
                solver.G_g, solver.G_f,
                x1_clean.to(device),
                n_show=10,
                title=f"Cycle: male x1 -> G_g(x1) -> G_f(G_g(x1))",
                row1_label="male $x_{1}$",
                row2_label="female? $G_g(x_{1})$",
                row3_label="male rec $G_f(G_g(x_{1}))$"
            )
    
    solver.G_f.eval()
    solver.G_g.eval()

    x0_f, _ = next(iter(female_loader))
    x1_m, _ = next(iter(male_loader))
    x0 = x0_f.to(device)
    x1 = x1_m.to(device)

    print("Done.")




if __name__ == "__main__":
    run_celeba_female_to_male_experiment()