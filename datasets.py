import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# =========================================================
# 3) CelebA female/male loaders
# =========================================================
def make_celeba_gender_loaders(
    root="./data", batch_size=64,
    num_workers=4, img_size=64,
    split="train", download=False,
):
    tfm = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeba_root = os.path.join(root, "celeba")
    print(f'CelebA root: {celeba_root}')
    required_files = [
        os.path.join(celeba_root, "identity_CelebA.txt"),
        os.path.join(celeba_root, "list_attr_celeba.txt"),
        os.path.join(celeba_root, "list_bbox_celeba.txt"),
        os.path.join(celeba_root, "list_eval_partition.txt"),
        os.path.join(celeba_root, "list_landmarks_align_celeba.txt"),
    ]
    image_dir = os.path.join(celeba_root, "img_align_celeba")

    if not download:
        missing = [p for p in required_files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "CelebA annotation files are missing.\n"
                f"Expected under: {celeba_root}\n"
                f"Missing files:\n" + "\n".join(missing)
            )

        if not os.path.isdir(image_dir):
            raise FileNotFoundError(
                "CelebA image folder is missing.\n"
                f"Expected folder: {image_dir}\n"
                "Please manually download and extract img_align_celeba.zip "
                "into ./data/celeba/, then set download=False."
            )

    ds = datasets.CelebA(
        root=root,
        split=split,
        target_type="attr",
        download=download,
        transform=tfm,
    )

    gender_idx = ds.attr_names.index("Male")

    male_indices = []
    female_indices = []

    # Faster than ds[i][1], because ds.attr is already loaded
    for i in range(len(ds)):
        if ds.attr[i, gender_idx].item() == 1:
            male_indices.append(i)
        else:
            female_indices.append(i)

    female_ds = Subset(ds, female_indices)
    male_ds = Subset(ds, male_indices)

    female_loader = DataLoader(
        female_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    male_loader = DataLoader(
        male_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    return female_loader, male_loader


def denorm(x):
    # x in [-1,1] -> [0,1]
    return (x * 0.5 + 0.5).clamp(0, 1)

@torch.no_grad()
def visualize_map(G, x_in, n_show=10, title="", in_label="input", out_label="mapped"):
    """
    Visualize mapping: x_in -> G(x_in)
    G: any nn.Module
    x_in: (B,3,H,W) in [-1,1]
    """
    G.eval()
    x_in = x_in[:n_show]

    x_out = G(x_in)

    x_in_vis  = (x_in  * 0.5 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()
    x_out_vis = (x_out * 0.5 + 0.5).clamp(0, 1).cpu().permute(0,2,3,1).numpy()

    plt.figure(figsize=(2*n_show, 4))
    for i in range(n_show):
        plt.subplot(2, n_show, i+1)
        plt.imshow(x_in_vis[i]); plt.axis("off")
        if i == 0: plt.ylabel(in_label)

        plt.subplot(2, n_show, n_show+i+1)
        plt.imshow(x_out_vis[i]); plt.axis("off")
        if i == 0: plt.ylabel(out_label)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def visualize_cycle(G_ab, G_ba, x_a, n_show=10, title="Cycle visualization",
                    row1_label="x_a", row2_label="G_ab(x_a)", row3_label="G_ba(G_ab(x_a))"):
    """
    Show: x_a -> G_ab(x_a) -> G_ba(G_ab(x_a))
    """
    G_ab.eval()
    G_ba.eval()

    x_a = x_a[:n_show]
    x_b = G_ab(x_a)
    x_a_rec = G_ba(x_b)

    def denorm01(z):
        return (z * 0.5 + 0.5).clamp(0, 1)

    row1 = denorm01(x_a).cpu().permute(0,2,3,1).numpy()
    row2 = denorm01(x_b).cpu().permute(0,2,3,1).numpy()
    row3 = denorm01(x_a_rec).cpu().permute(0,2,3,1).numpy()

    plt.figure(figsize=(2*n_show, 6))
    for i in range(n_show):
        plt.subplot(3, n_show, i+1)
        plt.imshow(row1[i]); plt.axis("off")
        if i == 0: plt.ylabel(row1_label, fontsize=12)

        plt.subplot(3, n_show, n_show+i+1)
        plt.imshow(row2[i]); plt.axis("off")
        if i == 0: plt.ylabel(row2_label, fontsize=12)

        plt.subplot(3, n_show, 2*n_show+i+1)
        plt.imshow(row3[i]); plt.axis("off")
        if i == 0: plt.ylabel(row3_label, fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    # plt.show()
    # save figure
    if os.path.exists("./samples") == False:
        os.makedirs("./samples")
    plt.savefig(f"./samples/{title.replace(' ','_')}.png")
    plt.close()


def visualize_mccann_interpolation(G_f, x0, n_show=10, lam=0.5, title="McCann interpolation"):
    """
    G_f: trained map, T(x)=G_f(x)  (no t-conditioning version)
    x0 : (B,3,H,W) in [-1,1]
    """
    G_f.eval()
    x0 = x0[:n_show]

    Tx0 = G_f(x0)
    x_mid = (1 - lam) * x0 + lam * Tx0

    def denorm(img):
        return (img * 0.5 + 0.5).clamp(0, 1)

    row1 = denorm(x0).detach().cpu().permute(0,2,3,1).numpy()
    row2 = denorm(x_mid).detach().cpu().permute(0,2,3,1).numpy()
    row3 = denorm(Tx0).detach().cpu().permute(0,2,3,1).numpy()

    plt.figure(figsize=(2*n_show, 6))
    for i in range(n_show):
        plt.subplot(3, n_show, i+1)
        plt.imshow(row1[i]); plt.axis("off")
        if i == 0: plt.ylabel("x0", fontsize=12)

        plt.subplot(3, n_show, n_show+i+1)
        plt.imshow(row2[i]); plt.axis("off")
        if i == 0: plt.ylabel(f"{lam}·T(x0)+(1-{lam})·x0", fontsize=12)

        plt.subplot(3, n_show, 2*n_show+i+1)
        plt.imshow(row3[i]); plt.axis("off")
        if i == 0: plt.ylabel("T(x0)", fontsize=12)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()