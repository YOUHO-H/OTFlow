import sys
from collections import namedtuple
from tqdm import trange
import yaml
from absl import app, flags
from utils.experiment import prepare
import torch
from data.cell import load_cell_data
from data.utils import cast_loader_to_iterator
from solvers.solvers_cell import BarycenterFlowSolver, get_schedules
Pair = namedtuple("Pair", "source target")

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("config", "", "Path to config")
flags.DEFINE_string("exp_group", "cellot_exps", "Name of experiment.")
flags.DEFINE_string("online", "offline", "Run experiment online or offline.")
flags.DEFINE_boolean("restart", False, "Delete cache.")
flags.DEFINE_boolean("debug", False, "Debug mode.")
flags.DEFINE_boolean("dry", False, "Dry mode.")
flags.DEFINE_boolean("verbose", False, "Run in verbose mode.")

def run_sell_ot(argv):

    print("\n=== Running Experiment: Cells ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    config, outdir = prepare(argv)
    loader = load_cell_data(config)
    iterator = cast_loader_to_iterator(loader, cycle_all=True)
    n_iters = config.training.n_iters
    step = 0

    ticker = trange(step, n_iters, initial=step, total=n_iters)
    solver = BarycenterFlowSolver(device, config)

    g_loss_acc, d_loss_acc, ot_loss_acc= 0, 0, 0
    g_test_acc, ot_test_acc, mmd_test_acc = 0, 0, 0
    for step in ticker:

        iterator_train_target = iterator.train.target
        iterator_train_source = iterator.train.source
        iterator_test_target = iterator.test.target
        iterator_test_source = iterator.test.source

        target = next(iterator_train_target).to(device)
        source = next(iterator_train_source).to(device)

        lambdas = get_schedules(step // 1000)
        weights = lambdas[0:3]

        lg, ld, lot= solver.train_step(source, target, weights)

        g_loss_acc += lg
        d_loss_acc += ld
        ot_loss_acc += lot

        test_target = next(iterator_test_target).to(device)
        test_source = next(iterator_test_source).to(device)
        test_lg, test_lot, test_mmd = solver.evaluate_step(test_source, test_target)
        g_test_acc += test_lg
        ot_test_acc += test_lot
        mmd_test_acc += test_mmd

        if step % 100 == 0:
            ticker.set_postfix({
                'G': f"{g_loss_acc/100:.3f}",
                'D': f"{d_loss_acc/100:.3f}",
                'OT': f"{ot_loss_acc/100:.3f}",
                'G_test': f"{g_test_acc/100:.3f}",
                'OT_test': f"{ot_test_acc/100:.3f}",
                'MMD_test': f"{mmd_test_acc/100:.3f}"
            })
            g_loss_acc, d_loss_acc, ot_loss_acc = 0, 0, 0   
            g_test_acc, ot_test_acc, mmd_test_acc = 0, 0, 0



if __name__ == "__main__":
    run_sell_ot(sys.argv)