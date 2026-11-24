# PROJECT: Training Pipeline

# TODO:
# - Make trainer and pipeline modular
#     - Current configuration is standard
#     - Modular Pipeline means design custom pipelines and checks/callbacks to manage and stop epochs as well as
#     optimizers and schedulers
#     - Modular Trainer means change it's component. It can be set up with custom loss function, custom optimizer, custom
#     scheduler

# DONE:
# - Move to CUDA-compatible code
#     - get_device() checks wheter CUDA is available and use it in every tensor in the model (see tools.py)
# - Optimize code speed (in-place operations wherever possible, all CUDA-compatible)
# - Move from an absolute min_loss_delta to a relative one
#     - A low absolute delta is ineffective early in training, when loss is high and the model is exploring the solution space
#     - A low delta is needed near the end of training, when loss is low, so the relative threshold scales with the loss
# - Compute the sign cross-talk matrix computationally
#     - cross-talk matrix valid for any N-mode photonic chips with Clements layout.
# - Check if model is properly reset after each restart
# - Study cross-talk checking and fixing frequency
#     - Observed large jumps in loss evolution when the cross-talk matrix is checked and fixed
#     - Investigate whether checking and fixing it more often improves convergence

from __future__ import annotations

import copy
import math
import pathlib
import pickle
import time
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

from calippo.tools import (
    amplitude_fidelity,
    clements_xtalk_sign,
    make_real_border,
    sinkhorn,
)

_SIGN_CACHE: dict = {}


@dataclass
class TrainingConfig:

    # Multi-Start Params
    train_ratio: float = 0.8
    current_2pi: float = 10.0
    target_fidelity: float = 0.9999
    num_starts: int = 10
    max_global_epochs: int = 1_000_000
    check_interval: int = 250

    # Stagnation & Escape Params
    escape_patience: int = 500
    max_escape_attempts: int = 15
    min_loss_rel_delta: float = 0.01
    xt_check_interval: int = 1000  # epochs counted from the start of each restart; 0 = off

    # Optimization
    initial_lr: float = 1e-7
    lr_scheduler_patience: int = 200
    lr_scheduler_factor: float = 0.5


def _xtalk_sign(model) -> torch.Tensor:
    """Cross-talk sign pattern as a tensor living on the model's device (built once, reused every epoch)."""
    key = (model.xtalk.device, model.xtalk.dtype)
    sign = _SIGN_CACHE.get(key)
    if sign is None:
        sign = clements_xtalk_sign(model.n_modes, dtype=model.xtalk.dtype).to(model.xtalk.device)
        _SIGN_CACHE[key] = sign
    return sign


def apply_constraints(model):
    # in-line updates
    with torch.no_grad():
        model.xtalk.abs_().mul_(_xtalk_sign(model))
        if model.non_linear:  # for a linear model nl_factor is identically zero
            model.nl_factor.abs_()


def state_dict_to_cpu(model):
    """
    Detached CPU copy of the model state_dict. Keeping checkpoints on the CPU makes the saved pickle
    loadable on machines without CUDA, and does not hold GPU memory. `load_state_dict` copies the
    tensors back onto whatever device the model lives on.
    """
    sd = model.state_dict()
    return type(sd)((k, v.detach().cpu().clone()) for k, v in sd.items())


def _plot_lr_finder(lrs, losses, optimal_lr, optimal_loss, noise_idx):
    """Plot the LR range test curve with the optimal and noise-onset points marked."""
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, label='Raw Loss', color='blue')
    plt.scatter(
        optimal_lr,
        optimal_loss,
        color='red',
        s=50,
        zorder=5,
        label=f'Optimal LR: {optimal_lr:.2e}',
    )
    plt.scatter(
        lrs[noise_idx],
        losses[noise_idx],
        color='orange',
        s=50,
        zorder=4,
        label='First Minimum (Noise Start)',
        marker='X',
    )
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss (MSE)')
    plt.title('Learning Rate Finder (Pre-Noise Steepest Descent)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.legend()
    plt.show()


def lr_finder(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    i_train: torch.Tensor,
    u_target_train: torch.Tensor,
    init_lr: float = 1e-7,
    max_lr: float = 1.0,
    steps: int = 100,
    plot_result: bool = True,
) -> float:
    """
    Classic LR range test: ramp the learning rate exponentially from
    `init_lr` to `max_lr` over `steps` optimizer steps, and pick the LR at
    the steepest point of descent *before* the loss starts climbing back up
    (which marks the onset of the noise-dominated regime).

    The model/optimizer state is restored to its pre-test values before
    returning, so this is a non-destructive probe.
    """
    initial_state = copy.deepcopy(model.state_dict())
    initial_optim = copy.deepcopy(optimizer.state_dict())

    param_group = optimizer.param_groups[0]
    param_group['lr'] = init_lr
    mult = (max_lr / init_lr) ** (1 / steps)

    lrs = []
    losses = []
    best_loss = math.inf

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(i_train), u_target_train)
        loss_val = loss.item()  # single host-device sync per step

        # Abort the sweep early if the loss has diverged.
        if loss_val > 10 * best_loss or torch.isnan(loss):
            break

        best_loss = min(best_loss, loss_val)

        loss.backward()
        optimizer.step()
        apply_constraints(model)

        lrs.append(param_group['lr'])
        losses.append(loss_val)
        param_group['lr'] *= mult

    loss_grads = np.gradient(losses)

    # First index (after a 5-step warm-up) where the loss gradient turns
    # positive: the loss starts climbing again and noise dominates.
    rising = np.flatnonzero(loss_grads[5:] > 0)
    noise_idx = int(rising[0]) + 5 if rising.size else len(losses) - 1

    # Steepest descent before the noise onset (signal-dominated regime).
    steepest_idx = 0 if noise_idx < 2 else int(np.argmin(loss_grads[:noise_idx]))

    optimal_lr = lrs[steepest_idx]
    optimal_loss = losses[steepest_idx]

    # Restore the model/optimizer exactly as they were before the sweep.
    model.load_state_dict(initial_state)
    model.M = model._compute_mixing_layer()
    optimizer.load_state_dict(initial_optim)

    if plot_result:
        _plot_lr_finder(lrs, losses, optimal_lr, optimal_loss, noise_idx)

    return optimal_lr


def reinitialize_model_parameters(model, current_2pi: float) -> None:
    """Reinitializes photonic processor static phases randomly in [0, 2π) and resets
    the cross-talk matrix to ideal diagonal coupling, applying structural masks.
    Sigma is set to π/4 everywhere.
    """
    with torch.no_grad():
        # Random uniform phases in [0, 2π) to explore diverse basins
        random_phases = torch.rand(model.n_phases, device=model.device, dtype=model.static_phases.dtype) * 2 * torch.pi
        model.static_phases.copy_(random_phases)

        # Reset cross-talk matrix to ideal (diagonal-only) scaled to produces a 2*pi phase shift with `current_2pi`
        ideal_xtalk = torch.eye(
            model.n_phases,
            device=model.xtalk.device,
            dtype=torch.float64,
        )
        model.xtalk.copy_(ideal_xtalk)
        model.xtalk.mul_(2 * torch.pi / (current_2pi**2))

        # Mask the output
        model.static_phases.mul_(model.static_phases_mask)
        model.xtalk.mul_(model.xtalk_mask)

        # Reset all beam splitters to π/4
        model.sigma.fill_(torch.pi / 4)
        model.M = model._compute_mixing_layer()
        if model.unbalanced_beam_splitters:
            model.sigma.requires_grad = False

        # Reset non-linear factor
        if model.non_linear:
            model.nl_factor.zero_()
            model.nl_factor.requires_grad = False


class CrosstalkManager:
    DIAG_TOLERANCE = 0.50  # max deviation of a diagonal entry from the reference diagonal
    OFFDIAG_LIMIT = 0.30  # max off-diagonal entry, as a fraction of the reference diagonal
    OFFDIAG_RESET = 0.10  # value an offending off-diagonal is reset to

    @staticmethod
    def analyze_anomalies(model):
        with torch.no_grad():
            used = model.xtalk_mask.bool()
            diag_used = used.diagonal()
            diag = model.xtalk.diagonal().abs()
            ref_diag = diag[diag_used].median().clamp_min(1e-8) if diag_used.any() else diag.new_tensor(1e-8)
            bad_diag = diag_used & ((diag - ref_diag).abs() > CrosstalkManager.DIAG_TOLERANCE * ref_diag)
            offdiag_used = used & ~torch.eye(used.shape[0], dtype=torch.bool, device=used.device)
            bad_off = offdiag_used & (model.xtalk.abs() > CrosstalkManager.OFFDIAG_LIMIT * ref_diag)
            return bad_diag, bad_off, ref_diag

    @staticmethod
    def reset_anomalies(model, bad_diag_mask, bad_offdiag_mask, ref_diag):
        with torch.no_grad():
            sign = _xtalk_sign(model)
            model.xtalk[bad_offdiag_mask] = sign[bad_offdiag_mask] * CrosstalkManager.OFFDIAG_RESET * ref_diag
            model.xtalk.diagonal()[bad_diag_mask] = sign.diagonal()[bad_diag_mask] * ref_diag
            sp_mask = bad_diag_mask & model.static_phases_mask.bool()
            n = sp_mask.sum().item()
            if n > 0:
                model.static_phases[sp_mask] = (
                    torch.rand(n, device=model.static_phases.device, dtype=torch.float64) * 2 * torch.pi
                )


class Trainer:
    def __init__(self, model, config: TrainingConfig):
        self.model = model
        self.config = config
        self.criterion = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

        self.init_optimizer(lr=1e-7)

    def init_optimizer(self, lr: float):
        params = list(self.model.parameters())
        try:
            # Fused Adam: one kernel for all parameters instead of several per parameter (CUDA only).
            fused = {"fused": True} if self.model.device.type == "cuda" else {}
            self.optimizer = torch.optim.Adam(params, lr=lr, **fused)
        except (RuntimeError, ValueError):
            self.optimizer = torch.optim.Adam(params, lr=lr)
        self.optimizer.state.clear()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=self.config.lr_scheduler_factor, patience=self.config.lr_scheduler_patience
        )

    def train_one_epoch(self, i_train, u_train) -> float:
        if self.optimizer is None or self.scheduler is None:
            raise ValueError(
                "Set the optimizer and scheduler before proceeding with training:\n",
                f"Optimizer = {self.optimizer}, Scheduler = {self.scheduler}",
            )

        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(i_train)
        loss = self.criterion(pred, u_train)

        loss.backward()
        self.optimizer.step()
        apply_constraints(self.model)

        loss_val = loss.item()
        self.scheduler.step(loss_val)
        return loss_val

    def evaluate(self, i_test, u_test) -> float:
        self.model.eval()
        with torch.no_grad():
            pred_test = self.model(i_test)
            fid = amplitude_fidelity(u_test, pred_test).mean().item()
        return fid


class TrainingPipeline:
    def __init__(self, trainer: Trainer, config: TrainingConfig):
        self.trainer = trainer
        self.config = config
        self.best_overall_fidelity = -1.0
        self.best_weights = None
        self.loss_history = {"epoch": [], "loss": []}
        self.eval_history = {"epoch": [], "fid_train": [], "fid_test": [], "lr": []}

    def run_multi_start(self, train_data, test_data):
        i_train, u_train = train_data
        i_test, u_test = test_data

        # Re-init and Find LR
        reinitialize_model_parameters(self.trainer.model, self.config.current_2pi)
        opt_lr = lr_finder(
            model=self.trainer.model,
            optimizer=self.trainer.optimizer,
            criterion=self.trainer.criterion,
            i_train=i_train,
            u_target_train=u_train,
        )
        self.trainer.init_optimizer(opt_lr)

        global_epoch = 0
        while global_epoch < self.config.max_global_epochs:
            for start_idx in range(self.config.num_starts):
                print(f" START {start_idx + 1}/{self.config.num_starts} ".center(120, '-'))

                # Start Training Loop
                success, epochs_spent, best_fid = self._execute_start(
                    i_train, u_train, i_test, u_test, opt_lr, current_epoch=global_epoch
                )
                global_epoch += epochs_spent

                if success:
                    print(f"Target fidelity achieved in start {start_idx+1}!\n")
                    break
                else:
                    print(f" [EOS] Best achieved fidelity: {best_fid:.4f} ".center(120, '-'), "\n")

                # Re-init and Find LR
                reinitialize_model_parameters(self.trainer.model, self.config.current_2pi)
                opt_lr = lr_finder(
                    model=self.trainer.model,
                    optimizer=self.trainer.optimizer,
                    criterion=self.trainer.criterion,
                    i_train=i_train,
                    u_target_train=u_train,
                    plot_result=False,
                )
                self.trainer.init_optimizer(opt_lr)
            break

    def _repair_crosstalk(self, restart_lr: float | None = None) -> int:
        """Look for anomalous cross-talk coefficients and reset them (bad diagonals also get new random static
        phases). Returns the number of ill elements found, 0 meaning that nothing was touched.

        ``restart_lr``: if given, Adam is restarted from this learning rate (fresh moments and scheduler). The periodic
        check does not need it: the optimizer state is still valid, and restarting it costs more than it brings.
        """
        model = self.trainer.model
        bad_diag, bad_off, mean_diag = CrosstalkManager.analyze_anomalies(model)
        n_ill = int(bad_diag.sum() + bad_off.sum())
        if n_ill:
            CrosstalkManager.reset_anomalies(model, bad_diag, bad_off, mean_diag)
            if restart_lr is not None:
                self.trainer.init_optimizer(restart_lr)
        return n_ill

    def _execute_start(self, i_train, u_train, i_test, u_test, opt_lr, current_epoch) -> tuple[bool, int, float]:
        best_loss_this_start = float('inf')
        best_fid_this_start = -1
        epochs_no_improve = 0
        escape_attempts = 0
        periodic_fixes = 0
        nl_frozen = True
        sigma_frozen = True
        old_epoch = current_epoch

        start_time = time.perf_counter()
        while current_epoch < self.config.max_global_epochs:
            current_epoch += 1
            loss = self.trainer.train_one_epoch(i_train, u_train)

            # Stagnation Tracking
            if loss < best_loss_this_start * (1.0 - self.config.min_loss_rel_delta):
                # if loss < best_loss_this_start - self.config.min_loss_delta:
                best_loss_this_start, epochs_no_improve = loss, 0
            else:
                epochs_no_improve += 1

            if self.config.xt_check_interval and (current_epoch - old_epoch) % self.config.xt_check_interval == 0:
                n_ill = self._repair_crosstalk()
                if n_ill:
                    periodic_fixes += 1
                    print(f"XT FIX [periodic] - Found {n_ill} ill-elements in cross-talk matrix.")
                    epochs_no_improve = 0
            # Evaluation & Curriculum learning only if not periodic xtalk fix occurred
            elif current_epoch % self.config.check_interval == 0 and current_epoch != 0:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                start_time = current_time

                fid_train = self.trainer.evaluate(i_train, u_train)
                fid_test = self.trainer.evaluate(i_test, u_test)
                print(
                    f" Global Epoch {current_epoch:7d} |",
                    f"LR: {self.trainer.optimizer.param_groups[0]['lr']:.2e} |",
                    f"Train Fid: {fid_train:.5f} |",
                    f"Test Fid: {fid_test:.5f} |",
                    f"Loss: {loss:.2e} |",
                    f"Time/Epoch: {elapsed_time * 1000 /self.config.check_interval:3.2f} ms",
                )

                self.eval_history["epoch"].append(current_epoch)
                self.eval_history["fid_train"].append(fid_train)
                self.eval_history["fid_test"].append(fid_test)
                self.eval_history["lr"].append(self.trainer.optimizer.param_groups[0]['lr'])

                self.loss_history["epoch"].append(current_epoch)
                self.loss_history["loss"].append(loss)

                if fid_test >= self.config.target_fidelity:
                    return True, current_epoch - old_epoch, fid_test
                if fid_test >= best_fid_this_start:
                    best_fid_this_start = fid_test

                # Curriculum learning: non-linear factor
                if self.trainer.model.non_linear and fid_test > 0.99 and nl_frozen:
                    print(" [ FID > 0.99 ] - Unfreezing phase shifters non-linear factors learning. ".center(120, '-'))
                    self.trainer.model.nl_factor.requires_grad = True
                    nl_frozen = False

                # Curriculum learning: unbalanced beam splitters
                if self.trainer.model.unbalanced_beam_splitters and fid_test > 0.98 and sigma_frozen:
                    print(" [ FID > 0.98 ] - Unfreezing splitting ratios learning. ".center(120, '-'))
                    self.trainer.model.sigma.requires_grad = True
                    sigma_frozen = False

                # Save best weights globally
                if fid_test > self.best_overall_fidelity:
                    self.best_overall_fidelity = fid_test
                    self.best_weights = {k: v.clone() for k, v in self.trainer.model.state_dict().items()}

            # Escape Logic
            if epochs_no_improve >= self.config.escape_patience:
                if escape_attempts >= self.config.max_escape_attempts:
                    print("Max escape attempts reached. Abandoning start.")
                    return False, current_epoch - old_epoch, best_fid_this_start

                # Check cross-talk matrix
                n_ill = self._repair_crosstalk(restart_lr=opt_lr)
                if n_ill:
                    escape_attempts += 1
                    epochs_no_improve = 0
                    print(
                        f"XT FIX [{escape_attempts}/{self.config.max_escape_attempts}] -",
                        f"Found {n_ill} ill-elements in cross-talk matrix.",
                    )
                else:
                    print("Stagnation detected but no anomalies found. Moving to next start.\n")
                    return False, current_epoch - old_epoch, best_fid_this_start

        return False, current_epoch - old_epoch, best_fid_this_start

    def plot_training_results(self, save_path: str | None = None, show: bool = False) -> plt.Figure:
        """
        Plot training loss, test and train fidelity, target fidelity, and learning rate.

        The figure has two panels sharing the epoch axis. The top panel shows train and
        test fidelity against the target fidelity, sampled every ``check_interval`` epochs.
        The bottom panel shows the per-epoch loss on a log scale, with the learning rate
        on a secondary log-scale axis.

        Args:
            save_path: Optional file path to save the figure, e.g. ``"training.png"``.
                The output format is inferred from the extension. If None, the figure
                is not saved.
            show: If True, display the figure with ``plt.show()``. Defaults to False so
                the method can be used in non-interactive environments.

        Returns:
            The matplotlib ``Figure`` object, which can be customized further or closed.

        Raises:
            ValueError: If no training history has been recorded. Call
                ``run_multi_start()`` before plotting.
        """

        if not self.loss_history["epoch"]:
            raise ValueError("No training history found. Run run_multi_start() first.")

        ev, lh = self.eval_history, self.loss_history
        fig, (ax_fid, ax_loss) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Fidelity panel
        ax_fid.plot(ev["epoch"], ev["fid_train"], label="Train fidelity")
        ax_fid.plot(ev["epoch"], ev["fid_test"], label="Test fidelity")
        ax_fid.axhline(self.config.target_fidelity, color="r", linestyle="--", label="Target")
        ax_fid.set_ylabel("Fidelity")
        ax_fid.grid(True, alpha=0.3)
        ax_fid.legend(loc="lower right")

        # Loss panel (log scale) with learning rate on a secondary axis
        ax_loss.semilogy(lh["epoch"], lh["loss"], color="tab:orange", label="Loss")
        ax_loss.set_ylabel("Loss (log scale)")
        ax_loss.set_xlabel("Global epoch")
        ax_loss.grid(True, alpha=0.3)

        ax_lr = ax_loss.twinx()
        ax_lr.semilogy(ev["epoch"], ev["lr"], color="tab:green", linestyle=":", label="Learning rate")
        ax_lr.set_ylabel("Learning rate (log scale)")

        lines, labels = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_lr.get_legend_handles_labels()
        ax_loss.legend(lines + lines2, labels + labels2, loc="upper right")

        fig.suptitle("Training Results")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)

        if show:
            plt.show()
        return fig


def load_data(
    data_filepath: str | pathlib.Path,
    train_ratio: float = 0.8,
    device: str | torch.device | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Load experimental data from disk.

    Parameters
    ----------
    data_filepath : str or pathlib.Path
        Path to the file containing the experimental data.
    train_ratio : float, default=0.8
        Fraction of the data to use for training.
    device : str or torch.device, optional
        Device on which parameters and buffers are stored. If None (default),
        CUDA is used when available, otherwise the CPU.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the training and test data. Each subset is a triple of
        PyTorch tensors containing the currents, processed ``U_raw``, and processed
        ``U_measured``, respectively.
        Order: (i_train, u_exp_train) and (i_test, u_exp_test)
    """
    print(f"Loading external experimental data from {data_filepath}...")
    with open(data_filepath, "rb") as f:
        data = pickle.load(f)
        u_th = data["u_the"]
        u_exp = data["u_exp"].transpose(1, 2)
        currents = data["currents"]

    total_samples = len(currents)
    n_train = int(total_samples * train_ratio)
    n_test = total_samples - n_train
    print(f"Total Samples: {total_samples} (Train: {n_train}, Test: {n_test}).")

    u_th, *_ = make_real_border(u_th)
    u_exp = torch.sqrt(sinkhorn(u_exp)[0]).abs()
    print(f"Fidelity baseline before training: {amplitude_fidelity(u_th.abs(), u_exp).mean():.4f}.\n")

    return (
        (currents[:n_train].to(device), u_exp[:n_train].to(device)),
        (currents[n_train:].to(device), u_exp[n_train:].to(device)),
    )


if __name__ == "__main__":
    from calippo.model import Clements
    from calippo.tools import get_device

    # Use the GPU if CUDA is available, otherwise fall back to the CPU
    device = get_device()
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")

    # Setup
    processor = Clements(
        n_modes=6,
        non_linear=False,
        unbalanced_beam_splitters=True,
        device=device,
    )
    cfg = TrainingConfig()
    datapath = "20261117_PIC6810.pkl"

    # Data Loading
    train_bundle, test_bundle = load_data(
        data_filepath=datapath,
        train_ratio=cfg.train_ratio,
        device=device,
    )

    # Training
    trainer = Trainer(processor, cfg)
    pipeline = TrainingPipeline(trainer, cfg)
    try:
        pipeline.run_multi_start(train_bundle, test_bundle)
    except KeyboardInterrupt:
        print("Ctrl-C program interrupted.\n")

    # Export best model weights
    val = f"{pipeline.best_overall_fidelity:.4f}".replace('.', '_')
    processor.export_calibration(f"{str(datapath).split('_')[0]}_F{val}.pkl", state_dict=pipeline.best_weights)

    # Plot training results
    pipeline.plot_training_results(save_path="train_results.png")
    # load model, evaluate on test set and plot fidelity histogram
