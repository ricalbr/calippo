from __future__ import annotations

import copy
import pathlib
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from calippo.tools import amplitude_fidelity, sinkhorn


# CONFIGURATION
@dataclass
class TrainingConfig:
    # Mesh Params
    n_modes: int = 6
    current_2pi: float = 17.0

    # Multi-Start Params
    train_ratio: float = 0.8
    num_starts: int = 10
    target_fidelity: float = 1 - 1e-4
    max_global_epochs: int = 1_000_000
    check_interval: int = 150

    # Stagnation & Escape Params
    escape_patience: int = 300
    max_escape_attempts: int = 5
    min_loss_delta: float = 1e-5

    # Optimization
    initial_lr: float = 1e-7
    lr_scheduler_patience: int = 100
    lr_scheduler_factor: float = 0.5


def lr_finder(
    model, optimizer, criterion, i_train, u_target_train, init_lr=1e-7, max_lr=1.0, steps=1000, plot_result=True
):
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

    mult = (max_lr / init_lr) ** (1 / steps)
    optimizer.param_groups[0]['lr'] = init_lr

    lrs = []
    losses = []
    best_loss = float('inf')

    model.train()
    for i in range(steps):
        optimizer.zero_grad()
        u_mag_pred = model(i_train)
        loss = criterion(u_mag_pred, u_target_train)

        # Abort the sweep early if the loss has diverged.
        if loss.item() > 10 * best_loss or torch.isnan(loss):
            break

        if loss.item() < best_loss:
            best_loss = loss.item()

        loss.backward()
        optimizer.step()

        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())

        optimizer.param_groups[0]['lr'] *= mult

    # Automated extraction
    loss_gradients = np.gradient(losses)

    # Find the first point (after a short warm-up of 5 steps) where the loss
    # curve's gradient turns positive: this is where noise starts to
    # dominate the LR sweep and the loss begins climbing again.
    first_min_idx = len(losses) - 1
    for i in range(5, len(loss_gradients)):
        if loss_gradients[i] > 0:
            first_min_idx = i
            break

    # The optimal LR is taken at the steepest descent *before* that point,
    # i.e. where the loss was falling fastest while still in the
    # signal-dominated (pre-noise) regime.
    if first_min_idx < 2:
        steepest_idx = 0
    else:
        steepest_idx = np.argmin(loss_gradients[:first_min_idx])

    optimal_lr = lrs[steepest_idx]

    # Restore the model/optimizer exactly as they were before the sweep.
    model.load_state_dict(initial_state)
    optimizer.load_state_dict(initial_optim)

    if plot_result:
        plt.figure(figsize=(8, 5))
        plt.plot(lrs, losses, label='Raw Loss', color='blue')
        plt.scatter(
            optimal_lr, losses[steepest_idx], color='red', s=50, zorder=5, label=f'Optimal LR: {optimal_lr:.2e}'
        )
        plt.scatter(
            lrs[first_min_idx],
            losses[first_min_idx],
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
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.show()

    return optimal_lr


def reinitialize_model_parameters(processor, current_2pi: float) -> None:
    """Reinitializes photonic processor static phases randomly in [0, 2π) and resets
    the cross-talk matrix to ideal diagonal coupling, applying structural masks.
    """
    with torch.no_grad():
        # Random uniform phases in [0, 2π) to explore diverse basins
        random_phases = torch.rand(processor.n_phases, dtype=processor.static_phases.dtype) * 2 * torch.pi
        processor.static_phases.copy_(random_phases)

        # Reset cross-talk matrix to ideal (diagonal-only) scaled to produces a 2*pi phase shift with `current_2pi`
        ideal_xtalk = torch.eye(
            processor.n_phases,
            dtype=processor.xtalk.dtype,
            device=processor.xtalk.device,
        )
        processor.xtalk.copy_(ideal_xtalk)
        processor.xtalk.mul_(2 * torch.pi / (current_2pi**2))

        # Mask the output
        processor.static_phases.mul_(processor.static_phases_mask)
        processor.xtalk.mul_(processor.xtalk_mask)

        if processor.nonlinear:
            processor.nl_factor.zero_()
            processor.nl_factor.requires_grad = False


# CROSSTALK MANAGER (Physics Logic)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, factor=self.config.lr_scheduler_factor, patience=self.config.lr_scheduler_patience
        )

    def train_one_epoch(self, i_train, u_mag_train) -> float:
        if self.optimizer is None or self.scheduler is None:
            raise ValueError(
                "Set the optimizer and scheduler before proceeding with training:\n",
                f"Optimizer = {self.optimizer}, Scheduler = {self.scheduler}",
            )

        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(i_train)
        loss = self.criterion(pred, u_mag_train)

        loss.backward()
        self.optimizer.step()
        self.scheduler.step(loss.detach())

        return loss.item()

    def evaluate(self, i_test, u_test) -> float:
        self.model.eval()
        with torch.inference_mode():
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
        for start_idx in range(self.config.num_starts):
            print(f" START {start_idx + 1}/{self.config.num_starts} ".center(100, '-'))
            while global_epoch < self.config.max_global_epochs:
                # Start Training Loop
                success, epochs_spent = self._execute_start(
                    i_train, u_mag_train, i_test, u_test, opt_lr, current_epoch=global_epoch
                )
                global_epoch += epochs_spent

                if success:
                    print(f"Target fidelity achieved in start {start_idx+1}!")
                    break

    def _execute_start(self, i_train, u_mag_train, i_test, u_test, opt_lr, current_epoch) -> tuple[bool, int]:
        best_loss_this_start = float('inf')
        epochs_no_improve = 0
        escape_attempts = 0
        nl_unfrozen = False

        epoch = current_epoch

        start_time = time.perf_counter()
        while epoch < self.config.max_global_epochs:
            epoch += 1
            loss = self.trainer.train_one_epoch(i_train, u_mag_train)

            # Stagnation Tracking
            if loss < best_loss_this_start - self.config.min_loss_delta:
                best_loss_this_start, epochs_no_improve = loss, 0
            else:
                epochs_no_improve += 1

            # Evaluation & Curriculum
            if epoch % self.config.check_interval == 0 and epoch != 0:
                current_time = time.perf_counter()
                elapsed_time = current_time - start_time
                start_time = current_time

                fid_test = self.trainer.evaluate(i_test, u_test)
                print(
                    f"Global Epoch {epoch:6d} |",
                    f"LR: {self.trainer.optimizer.param_groups[0]['lr']:.2e} |",
                    # f"Train Fid: {self.trainer.optimizer.lr:.5f} |",
                    f"Test Fid: {fid_test:.5f} |",
                    f"Loss: {loss:.2e} |",
                    f"Elapsed time: {elapsed_time:.2f} s",
                )

                if fid_test >= self.config.target_fidelity:
                    return True, epoch

                if self.trainer.model.nonlinear and fid_test > 0.99 and not nl_unfrozen:
                    print("\n[PHASE 2] High fidelity reached. Unfreezing nl_factor.")
                    self.trainer.model.nl_factor.requires_grad = True

                # Save best weights globally
                if fid_test > self.best_overall_fidelity:
                    self.best_overall_fidelity = fid_test
                    self.best_weights = {k: v.clone() for k, v in self.trainer.model.state_dict().items()}
            # Escape Logic
            if epochs_no_improve >= self.config.escape_patience:
                if escape_attempts >= self.config.max_escape_attempts:
                    print("Max escape attempts reached. Abandoning start.")
                    return False, epoch

                # Perform Targeted Escape
                bad_diag, bad_off, mean_diag = CrosstalkManager.analyze_anomalies(self.trainer.model)

                if bad_diag.any() or bad_off.any():
                    escape_attempts += 1
                    CrosstalkManager.reset_anomalies(self.trainer.model, bad_diag, bad_off, mean_diag)

                    # Reset Optimizer state after physical reset
                    self.trainer.init_optimizer(opt_lr)
                    epochs_no_improve = 0
                    print(f"Update cross-talk matrix. {np.sum(bad_diag) + np.sum(bad_off)} elements were off.")
                    print(f"Targeted Escape {escape_attempts}/{self.config.max_escape_attempts} applied.")
                else:
                    print("Stagnation detected but no anomalies found. Moving to next start.")
                    return False, epoch

        return False, self.config.max_global_epochs


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


def generate_synthetic_data(
    processor: torch.nn.Module,
    nonlinear: bool = False,
    n_modes: int | None = None,
    current_2pi: float | None = None,
    mock_samples: int = 10_000,
    train_ratio: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthesize mock data using a mesh model.

    Parameters
    ----------
    processor : torch.nn.Module
        Mesh model used to synthesize mock data.
    nonlinear : bool, default=False
        Whether to use the nonlinear model.
    n_modes : int, optional
        Number of modes to retain. If ``None``, all modes are used.
    current_2pi : float, optional
        Scaling factor applied to the currents. If ``None``, no scaling is applied.
    mock_samples : int, default=10000
        Number of mock samples to synthesize.
    train_ratio : float, default=0.8
        Fraction of the data to use for training.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple containing the currents and U_measured tensors.
    """
    print("Generating mock experimental data...")

    gt_processor = processor(n_modes=n_modes, nonlinear=nonlinear)

    currents = gt_processor.generate_random_currents(current_ampl=current_2pi, batch_size=mock_samples)
    with torch.no_grad():
        phases_exp = gt_processor.phases_from_currents(currents)
        u_exp = gt_processor.U_from_phases(phases_exp)

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
    processor = Clements(n_modes=cfg.n_modes, nonlinear=True)

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
