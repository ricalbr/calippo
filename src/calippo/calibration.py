from __future__ import annotations

import copy
import pathlib
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from calippo.tools import amplitude_fidelity, sinkhorn


@dataclass
class TrainingConfig:

    # Multi-Start Params
    train_ratio: float = 0.8
    current_2pi: float = 10.0
    num_starts: int = 10
    target_fidelity: float = 0.9999
    max_global_epochs: int = 1_000_000
    check_interval: int = 100

    # Stagnation & Escape Params
    escape_patience: int = 300
    max_escape_attempts: int = 5
    min_loss_delta: float = 1e-7

    # Optimization
    initial_lr: float = 1e-7
    lr_scheduler_patience: int = 100
    lr_scheduler_factor: float = 0.25


# fmt: off
sign_data = [
    [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -1, -1, -1, 1, -1, -1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
]
# fmt: on


def apply_constraints(model):
    with torch.no_grad():
        model.mesh.xtalk.data = model.mesh.xtalk.abs() * torch.from_numpy(np.array(sign_data))
        model.mesh.nl_factor.data = model.mesh.nl_factor.abs()


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
    model: nn.Module,
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
        random_phases = torch.rand(model.n_phases, dtype=model.static_phases.dtype) * 2 * torch.pi
        model.static_phases.copy_(random_phases)

        # Reset cross-talk matrix to ideal (diagonal-only) scaled to produces a 2*pi phase shift with `current_2pi`
        ideal_xtalk = torch.eye(
            model.n_phases,
            dtype=model.xtalk.dtype,
            device=model.xtalk.device,
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
        if model.nonlinear:
            model.nl_factor.zero_()
            model.nl_factor.requires_grad = False


class CrosstalkManager:
    """Handles detection and resetting of anomalous crosstalk coefficients."""

    @staticmethod
    def analyze_anomalies(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns masks for bad diagonal and bad off-diagonal entries."""
        with torch.no_grad():
            xt_active = model.xtalk * model.xtalk_mask
            diag = xt_active.diagonal()
            diag_mask = model.xtalk_mask.diagonal().bool()

            valid_diag = diag[diag_mask]
            mean_diag = valid_diag.abs().mean().clamp_min(1e-8) if valid_diag.numel() > 0 else torch.tensor(1e-8)

            # Diagonal: Deviating > 50% from mean
            diag_dev = (diag.abs() - mean_diag).abs() / mean_diag
            bad_diag_mask = (diag_dev > 0.50) & diag_mask

            # Off-Diagonal: > 30% of mean diagonal
            offdiag_mask = model.xtalk_mask.bool().clone()
            offdiag_mask.fill_diagonal_(False)
            bad_offdiag_mask = offdiag_mask & (model.xtalk.abs() / mean_diag > 0.30)

            return bad_diag_mask, bad_offdiag_mask, mean_diag

    @staticmethod
    def reset_anomalies(model, bad_diag_mask, bad_offdiag_mask, mean_diag):
        """Applies resets to the mesh parameters."""
        with torch.no_grad():
            # Reset off-diagonals to a small fraction of mean
            model.xtalk[bad_offdiag_mask] = torch.sign(model.xtalk[bad_offdiag_mask]) * 0.10 * mean_diag

            # Reset diagonals to mean
            model.xtalk.diagonal()[bad_diag_mask] = torch.sign(model.xtalk.diagonal()[bad_diag_mask]) * mean_diag

            # Randomize phases for bad diagonals
            sp_mask = bad_diag_mask & model.static_phases_mask.bool()
            num_bad = sp_mask.sum().item()
            if num_bad > 0:
                model.static_phases[sp_mask] = torch.rand(num_bad, device=model.static_phases.device) * 2 * torch.pi


class Trainer:
    def __init__(self, model, config: TrainingConfig):
        # TODO: make it accept different losses, optimizers and schedulers
        self.model = model
        self.config = config
        self.criterion = torch.nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

        self.init_optimizer(lr=1e-7)

    def init_optimizer(self, lr: float):
        self.optimizer.state.clear()  # TODO this was not present in original pipeline, just in escape sequence. CHECK!!!
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
        self.scheduler.step(loss.detach())
        apply_constraints(calib_model)

        return loss.item()

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

    def run_multi_start(self, train_data, test_data):
        i_train, u_mag_train, _ = train_data
        i_test, _, u_test = test_data

        # Re-init and Find LR
        reinitialize_model_parameters(self.trainer.model, self.config.current_2pi)
        opt_lr = lr_finder(self.trainer.model, self.trainer.optimizer, self.trainer.criterion, i_train, u_mag_train)
        self.trainer.init_optimizer(opt_lr)

        global_epoch = 0
        while global_epoch < self.config.max_global_epochs:
            for start_idx in range(self.config.num_starts):
                print(f" START {start_idx + 1}/{self.config.num_starts} ".center(100, '-'))
                # Start Training Loop
                success, epochs_spent = self._execute_start(
                    i_train, u_mag_train, i_test, u_test, opt_lr, current_epoch=global_epoch
                )
                global_epoch += epochs_spent

                if success:
                    print(f"Target fidelity achieved in start {start_idx+1}!")
                    break

                # Re-init and Find LR
                opt_lr = lr_finder(
                    self.trainer.model, self.trainer.optimizer, self.trainer.criterion, i_train, u_mag_train
                )
                reinitialize_model_parameters(self.trainer.model, self.config.current_2pi)
                self.trainer.init_optimizer(opt_lr)
            break

    def _execute_start(self, i_train, u_mag_train, i_test, u_test, opt_lr, current_epoch) -> tuple[bool, int]:
        best_loss_this_start = float('inf')
        epochs_no_improve = 0
        escape_attempts = 0
        nl_frozen = True
        sigma_frozen = True
        old_epoch = current_epoch

        start_time = time.perf_counter()
        while current_epoch < self.config.max_global_epochs:
            current_epoch += 1
            loss = self.trainer.train_one_epoch(i_train, u_mag_train)

            # Stagnation Tracking
            if loss < best_loss_this_start - self.config.min_loss_delta:
                best_loss_this_start, epochs_no_improve = loss, 0
            else:
                epochs_no_improve += 1

            # Evaluation & Curriculum
            if current_epoch % self.config.check_interval == 0 and current_epoch != 0:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                start_time = current_time

                fid_test = self.trainer.evaluate(i_test, u_test)
                print(
                    f"Global Epoch {current_epoch:6d} |",
                    f"LR: {self.trainer.optimizer.param_groups[0]['lr']:.2e} |",
                    # f"Train Fid: {self.trainer.optimizer.lr:.5f} |",
                    f"Test Fid: {fid_test:.5f} |",
                    f"Loss: {loss:.2e} |",
                    f"Elapsed time: {elapsed_time:.2f} s",
                )

                if fid_test >= self.config.target_fidelity:
                    return True, current_epoch - old_epoch

                # Curriculum learning: non-linear factor
                if self.trainer.model.nonlinear and fid_test > 0.99 and nl_frozen:
                    print("Fidelity reached 99%. Unfreezing non-linear factor.")
                    self.trainer.model.nl_factor.requires_grad = True
                    nl_frozen = False

                # Curriculum learning: unbalanced beam splitters
                if self.trainer.model.learn_bs and fid_test > 0.98 and sigma_frozen:
                    print("Fidelity reached 98%. Unfreezing Beam Splitter learning.")
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
                    return False, current_epoch - old_epoch

                # Check cross-talk matrix
                bad_diag, bad_off, mean_diag = CrosstalkManager.analyze_anomalies(self.trainer.model)

                if bad_diag.any() or bad_off.any():
                    escape_attempts += 1
                    CrosstalkManager.reset_anomalies(self.trainer.model, bad_diag, bad_off, mean_diag)

                    # Reset Optimizer state after physical reset
                    self.trainer.init_optimizer(opt_lr)
                    epochs_no_improve = 0

                    print(f"Update cross-talk matrix. Found {torch.sum(bad_diag) + torch.sum(bad_off)} ill-elements.")
                    print(f"Targeted Escape {escape_attempts}/{self.config.max_escape_attempts} applied.")
                else:
                    print("Stagnation detected but no anomalies found. Moving to next start.")
                    return False, current_epoch - old_epoch

        return False, current_epoch - old_epoch


def load_data(
    data_filepath: str | pathlib.Path,
    train_ratio: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load experimental data from disk.

    Parameters
    ----------
    data_filepath : str or pathlib.Path
        Path to the file containing the experimental data.
    train_ratio : float, default=0.8
        Fraction of the data to use for training.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the currents and processed U_measured tensors.
    """
    print(f"Loading external experimental data from {data_filepath}...")

    data = np.load(data_filepath)
    currents = torch.from_numpy(data["currents"]).float()
    u_exp = torch.from_numpy(data["u_exp"])
    u_exp = torch.sqrt(sinkhorn(u_exp)[0])

    return split_train_test(currents, u_exp, train_ratio=train_ratio)


def split_train_test(
    currents: torch.Tensor,
    u_exp: torch.Tensor,
    train_ratio: float = 0.8,
) -> tuple:
    """Split data into training and test sets.

    Parameters
    ----------
    currents : torch.Tensor
        Input currents tensor.
    u_exp : torch.Tensor
        Processed U_measured tensor.
    train_ratio : float, default=0.8
        Fraction of the data to use for training.

    Returns
    -------
    tuple
        A tuple containing the training and test data. Each subset is a triple of
        PyTorch tensors containing the currents, processed ``U_raw``, and processed
        ``U_measured``, respectively.
    """
    # Amplitude matrix
    u_mag_exp = torch.abs(u_exp)

    # Dataset splitting
    total_samples = len(currents)
    n_train = int(total_samples * train_ratio)
    n_test = total_samples - n_train

    print(f"Total Samples: {total_samples} (Train: {n_train}, Test: {n_test})")

    return (
        (currents[:n_train], u_exp[:n_train], u_mag_exp[:n_train]),
        (currents[n_train:], u_exp[n_train:], u_mag_exp[n_train:]),
    )


if __name__ == "__main__":
    from calippo.model import Clements

    # Mesh Params
    cfg = TrainingConfig()

    # Initial Setup
    processor = Clements(
        n_modes=6,
        non_linear=False,
        unbalanced_beam_splitters=True,
    )

    # Data Loading
    train_bundle, test_bundle = load_data(
        data_filepath="20250626_dataset.npy.npz",
        train_ratio=cfg.train_ratio,
    )

    # Training
    trainer = Trainer(processor, cfg)
    pipeline = TrainingPipeline(trainer, cfg)
    try:
        pipeline.run_multi_start(train_bundle, test_bundle)
    except KeyboardInterrupt:
        print("Ctrl-C program interrupted.\n")

    # TODO: add visualization, model export
    # plot_training_history(pipeline.history, cfg.target_fidelity, path="training_history.png")

    processor.export_calibration("calibration_trained.pkl", state_dict=pipeline.best_weights)

    # TODO: check if traning config is really necessary, unpack all data
