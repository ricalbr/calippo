#!/usr/bin/env python3
"""
Multi-start calibration of a UPP mesh (Bell/Clements) via gradient descent
on the amplitude-only forward model, with a targeted "cross-talk escape"
mechanism to recover from stagnation.

Design summary
---------------
1. MULTI-START: each start reinitializes the mesh with random static
   phases and re-runs the automatic LR finder, so different starts explore
   different basins of the (non-convex) phase-retrieval landscape.
2. STAGNATION DETECTION: per-epoch, based on the absolute improvement of
   the raw training MSE loss (not a relative delta), so it isn't fooled by
   the LR shrinking over time via the scheduler.
3. TARGETED ESCAPE (not a blind random kick): when training stagnates, the
   cross-talk matrix is inspected for statistically anomalous diagonal
   (self heater-efficiency) or off-diagonal (cross-heater coupling)
   entries. Only those specific entries -- and their associated static
   phases -- are reset, while the rest of the fit is preserved. If no
   anomaly explains the stagnation, the start is abandoned in favor of a
   fresh one, since the current basin is assumed to be exhausted.
4. The best model (by test amplitude fidelity) across all starts and all
   escape attempts is kept and reloaded at the end.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import traceback
import time

from calippo.model import Bell, Clements
from calippo.tools import amplitude_fidelity, sinkhorn


class CalibrationModel(nn.Module):
    """
    Wrapper to train the mesh parameters while strictly enforcing
    the physical constraints (masks) during the forward pass.
    """

    def __init__(self, mesh_instance):
        super().__init__()
        self.mesh = mesh_instance

    def forward(self, currents):
        # Mask gradients to enforce physical parameters and prevent optimizer artifacts.
        # (the optimizer must not "invent" nonexistent coupling paths or heaters).
        sp_active = self.mesh.static_phases * self.mesh.static_phases_mask
        xt_active = self.mesh.xtalk * self.mesh.xtalk_mask

        # Phase model combining static offset and masked current-dependent cross-talk
        I2 = currents**2
        phases = sp_active + I2 @ xt_active.T

        # Native unitary propagation avoiding manual 2*pi wrapping to prevent gradient discontinuities.
        U = self.mesh.U_from_phases(phases)
        return torch.abs(U)


def lr_finder(
    model, optimizer, criterion, I_train, U_target_train, init_lr=1e-7, max_lr=10.0, steps=100, plot_result=False
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
        U_mag_pred = model(I_train)
        loss = criterion(U_mag_pred, U_target_train)

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

    # --- AUTOMATED EXTRACTION (raw loss) ---
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


def circular_abs_error(a, b, period=2 * np.pi):
    """
    Absolute error between two angles (or arrays of angles), correctly
    accounting for wrap-around: e.g. 0.01 rad and (2*pi - 0.01) rad are
    ~0.02 rad apart, not ~2*pi apart. Without this, comparing phases that
    have been wrapped into [0, 2*pi) with a plain `abs(a - b)` can report a
    near-perfect estimate as a near-worst-case error whenever the true and
    estimated phases straddle the 0 / 2*pi boundary.
    """
    diff = np.mod(a - b + period / 2, period) - period / 2
    return np.abs(diff)


def load_data(
    load_external: bool,
    data_filepath: str | None = None,
    train_ratio: float = 0.8,
    mesh_class: type | None = None,
    n_modes: int | None = None,
    current_2pi: float | None = None,
    mock_samples: int = 10_000,
):
    """Load experimental data from disk or synthesize mock data using a mesh model."""
    if load_external:
        if not data_filepath:
            raise ValueError("`data_filepath` is required when `load_external=True`.")

        print(f"Loading external experimental data from {data_filepath}...")
        with open(data_filepath, 'rb') as f:
            u_th = np.load(f)
            U_np_reshaped = np.load(f)
            currents = np.load(f)

        # Convert to PyTorch tensors and compute U amplitude
        I_exp = torch.from_numpy(currents).float()
        U_exp = torch.sqrt(sinkhorn(torch.from_numpy(U_np_reshaped).cdouble())[0])
    else:
        print("Generating mock experimental data...")
        gt_processor = mesh_class(n_modes=n_modes)

        I_exp = gt_processor.generate_random_currents(current_ampl=current_2pi, batch_size=total_samples)

        with torch.no_grad():
            phases_exp = gt_processor.phases_from_currents(I_exp)
            U_exp = gt_processor.U_from_phases(phases_exp)

    # Amplitude matrix
    U_mag_exp = torch.abs(U_exp)

    # Dataset splitting
    total_samples = len(I_exp)
    n_train = int(total_samples * train_ratio)
    n_test = total_samples - n_train

    print(f"Total Samples: {total_samples} (Train: {n_train}, Test: {n_test})")

    splits = (
        (I_exp[:n_train], I_exp[n_train:]),
        (U_exp[:n_train], U_exp[n_train:]),
        (U_mag_exp[:n_train], U_mag_exp[n_train:]),
    )

    return splits


def reinitialize_processor_parameters(processor, current_2pi: float) -> None:
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


def targeted_xtalk_escape(
    calib_model,
    optimizer,
    optimal_lr: float,
    scheduler,
    escape_attempts: int,
    max_escape_attempts: int,
    escape_patience: int,
    epochs_no_improve: int,
):
    """Inspects the cross-talk matrix for statistical anomalies (diagonal self-efficiency and
    off-diagonal coupling) and resets them along with corresponding static phases.

    Returns:
        Tuple containing updated (escape_attempts, epochs_no_improve, trigger_next_start,
        scheduler).
    """
    trigger_next_start = False
    scheduler = None

    # Use inference_mode to avoid tracking autograd operations during targeted reset
    with torch.inference_mode():
        mesh = calib_model.mesh
        xt_active = mesh.xtalk * mesh.xtalk_mask
        n_p = xt_active.shape[0]

        # Diagonal anomalies (self heater-efficiency)
        diag = xt_active.diagonal()
        diag_mask = mesh.xtalk_mask.diagonal().bool()

        valid_diag = diag[diag_mask]
        mean_diag = (
            valid_diag.abs().mean().clamp_min(1e-8)
            if valid_diag.numel() > 0
            else torch.tensor(1e-8, device=diag.device)
        )

        # Flag diagonal entries deviating >50% from mean self-efficiency
        diag_deviation = (diag.abs() - mean_diag).abs() / mean_diag
        bad_diag_mask = (diag_deviation > 0.50) & diag_mask

        # Off-diagonal anomalies (cross-heater coupling >30% of mean diagonal)
        offdiag_mask = mesh.xtalk_mask.bool().clone()
        offdiag_mask.fill_diagonal_(False)

        offdiag_ratio_2d = mesh.xtalk.abs() / mean_diag
        bad_offdiag_mask_2d = offdiag_mask & (offdiag_ratio_2d > 0.30)

        has_bad_diag = bad_diag_mask.any()
        has_bad_offdiag = bad_offdiag_mask_2d.any()

        if has_bad_diag or has_bad_offdiag:
            escape_attempts += 1

            # Reset anomalous off-diagonal entries
            if has_bad_offdiag:
                mesh.xtalk.data[bad_offdiag_mask_2d] = (
                    torch.sign(mesh.xtalk.data[bad_offdiag_mask_2d]) * 0.10 * mean_diag
                )

            # Reset anomalous diagonal entries & randomize static phases
            if has_bad_diag:
                diag_data = mesh.xtalk.data.diagonal()
                diag_data[bad_diag_mask] = torch.sign(diag_data[bad_diag_mask]) * mean_diag

                bad_sp_mask = bad_diag_mask & mesh.static_phases_mask.bool()
                num_bad_sp = bad_sp_mask.sum().item()

                if num_bad_sp > 0:
                    mesh.static_phases.data[bad_sp_mask] = (
                        torch.rand(num_bad_sp, dtype=mesh.static_phases.dtype, device=mesh.static_phases.device)
                        * 2
                        * torch.pi
                    )

            # Reset optimizer state to purge outdated momentum/second moments
            epochs_no_improve = 0
            optimizer.state.clear()
            optimizer.param_groups[0]["lr"] = optimal_lr
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50)

            num_bad_diag = bad_diag_mask.sum().item()
            num_bad_offdiag = bad_offdiag_mask_2d.sum().item()

            print(
                f"\n[TARGETED XTALK ESCAPE] Reset {num_bad_diag} diagonal (with static phases) "
                f"and {num_bad_offdiag} off-diagonal anomalous active coefficients "
                f"(Attempt {escape_attempts}/{max_escape_attempts})."
            )
        else:
            print(
                f"\nLoss stagnated for {escape_patience} epochs, but no anomalous xtalk found. "
                "Triggering next start..."
            )
            trigger_next_start = True

    return escape_attempts, epochs_no_improve, trigger_next_start, scheduler


@torch.inference_mode()
def apply_xtalk_bounds(mesh: torch.nn.Module) -> None:
    """Applies physical bounds to cross-talk matrix parameters based on mean heater efficiency.

    Diagonal self-efficiency entries are clamped within 50% of their mean
    magnitude. Active off-diagonal coupling terms are clamped within 30% of the
    mean diagonal magnitude.
    """
    # Extract diagonal view and active mask
    # Extract zero-copy diagonal view and active mask
    diag = mesh.xtalk.diagonal()  # type: ignore[attr-defined]
    diag_mask = mesh.xtalk_mask.diagonal().bool()  # type: ignore[attr-defined]

    valid_diag = diag[diag_mask]
    if valid_diag.numel() == 0:
        return

    # Compute mean self-efficiency magnitude
    mean_diag = valid_diag.abs().mean().clamp_min(1e-8)

    # Clamp active diagonal entries preserving sign: [0.50 * mean, 1.50 * mean]
    min_diag_mag = 0.50 * mean_diag
    max_diag_mag = 1.50 * mean_diag

    diag_sign = torch.sign(diag)
    # Handle exact zero signs by preserving positive magnitude default
    diag_sign[diag_sign == 0] = 1.0

    clamped_diag_mag = diag.abs().clamp(min=min_diag_mag, max=max_diag_mag)
    diag.copy_(diag_sign * clamped_diag_mag)

    # 2. Clamp active off-diagonal magnitudes preserving sign: [0, 0.30 * mean]
    offdiag_mask = mesh.xtalk_mask.bool().clone()  # type: ignore[attr-defined]
    offdiag_mask.fill_diagonal_(False)

    max_offdiag_mag = 0.30 * mean_diag
    offdiag_data = mesh.xtalk.data[offdiag_mask]  # type: ignore[attr-defined]

    offdiag_sign = torch.sign(offdiag_data)
    offdiag_sign[offdiag_sign == 0] = 1.0

    clamped_offdiag_mag = offdiag_data.abs().clamp(max=max_offdiag_mag)
    mesh.xtalk.data[offdiag_mask] = offdiag_sign * clamped_offdiag_mag  # type: ignore[attr-defined]


def run_start_diagnostics(
    start_idx: int,
    start_failed: bool,
    final_test_fid: float,
    calib_model: torch.nn.Module,
    circular_abs_error,
    gt_processor: torch.nn.Module | None = None,
    plot_per_start: bool = False,
) -> None:
    """Logs parameter estimation diagnostics and plots errors or estimated maps.
    Evaluates both ground-truth comparison (if available) and experimental data
    parameter visualisations.
    """
    status_str = "Failed to reach target" if start_failed else "Successfully reached target"
    print(f"\n[DIAGNOSTICS] Start {start_idx + 1} {status_str} " f"(Final Test Fid: {final_test_fid:.5f}).")

    with torch.inference_mode():
        mesh = calib_model.mesh  # type: ignore[attr-defined]
        est_sp = np.mod((mesh.static_phases * mesh.static_phases_mask).cpu().numpy(), 2 * np.pi)
        est_xt = (mesh.xtalk * mesh.xtalk_mask).cpu().numpy()

    if gt_processor is not None:
        print("Analyzing parameter errors against Ground Truth...")
        with torch.inference_mode():
            gt_sp = np.mod((gt_processor.static_phases * gt_processor.static_phases_mask).cpu().numpy(), 2 * np.pi)
            gt_xt = (gt_processor.xtalk * gt_processor.xtalk_mask).cpu().numpy()

            sp_mask_np = gt_processor.static_phases_mask.cpu().numpy().astype(bool)
            xt_mask_np = gt_processor.xtalk_mask.cpu().numpy() > 0

        sp_error = circular_abs_error(est_sp, gt_sp)
        xt_error = np.abs(est_xt - gt_xt)

        print(
            f"  -> Static Phases [0, 2pi] - Max Circular Error: {np.max(sp_error):.4e} | "
            f"Mean Error: {np.mean(sp_error[sp_mask_np]):.4e}"
        )
        print(
            f"  -> Cross-Talk Matrix - Max Absolute Error: {np.max(xt_error):.4e} | "
            f"Mean Error: {np.mean(xt_error[xt_mask_np]):.4e}"
        )

        worst_sp_idx = int(np.argmax(sp_error))
        print(
            f"  -> Worst Static Phase index: {worst_sp_idx} "
            f"(Est: {est_sp[worst_sp_idx]:.4f}, GT: {gt_sp[worst_sp_idx]:.4f})"
        )

        worst_xt_idx = np.unravel_index(np.argmax(xt_error), xt_error.shape)
        print(
            f"  -> Worst Cross-Talk coordinates: {worst_xt_idx} "
            f"(Est: {est_xt[worst_xt_idx]:.4f}, GT: {gt_xt[worst_xt_idx]:.4f})"
        )

        if plot_per_start:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            im0 = axes[0].imshow(est_xt - gt_xt, cmap="coolwarm", aspect="auto")
            axes[0].set_title(f"Start {start_idx + 1}: Cross-Talk Error (Est - GT)")
            fig.colorbar(im0, ax=axes[0])

            axes[1].plot(gt_sp, label="GT Static Phases", marker="o", alpha=0.7)
            axes[1].plot(est_sp, label="Est Static Phases [0, 2pi]", marker="x", alpha=0.7)
            axes[1].set_title(f"Start {start_idx + 1}: Static Phases Comparison")
            axes[1].legend()
            axes[1].grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            plt.show()

    else:
        if plot_per_start:
            print("Plotting estimated parameters (No Ground Truth available for experimental data)...")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            im0 = axes[0].imshow(est_xt, cmap="viridis", aspect="auto")
            axes[0].set_title(f"Start {start_idx + 1}: Estimated Cross-Talk Matrix")
            fig.colorbar(im0, ax=axes[0])

            axes[1].plot(est_sp, label="Est Static Phases [0, 2pi]", marker="x", color="orange")
            axes[1].set_title(f"Start {start_idx + 1}: Estimated Static Phases")
            axes[1].legend()
            axes[1].grid(True, linestyle=":", alpha=0.6)

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":

    calib_model = None
    gt_processor = None

    mesh_choice = "Clements"
    MeshClass = Bell if mesh_choice == "Bell" else Clements

    n = 6
    train_ratio = 0.8
    current_2pi = 17  # mA

    load_external_data = True
    data_filepath = "unitary_measurement_20260121_1.npy"

    # --- Multi-start configuration ---
    num_starts = 10  # max number of independent random restarts
    target_fidelity = 0.9985  # stop as soon as a start reaches this test fidelity
    check_interval = 100  # epochs between test-set evaluations
    max_global_epochs = 100_000  # hard cap on epochs summed across ALL starts

    # --- Stagnation & targeted escape configuration ---
    escape_patience = 200  # epochs with no raw-loss improvement before an escape is considered
    max_escape_attempts = 5  # escapes allowed within a single start before giving up on it
    min_loss_delta = 1e-6  # minimum absolute MSE improvement to NOT count as stagnation

    # --- Diagnostics ---
    plot_lr_finder = False  # Show LR-finder plot only on first start by default to avoid notebook clutter
    plot_per_start = False  # Show per-start diagnostic plot

    # --- Data Loading ---
    (I_train, I_test), (U_train, U_test), (U_mag_train, U_mag_test) = load_data(
        load_external=True,
        data_filepath=data_filepath,
        mesh_class=MeshClass,
        n_modes=n,
        current_2pi=current_2pi,
        train_ratio=train_ratio,
    )

    # MULTI-START OPTIMIZATION LOOP
    best_overall_fidelity = -1.0
    best_model_weights = None
    interrupted = False  # set to True on a user-requested Ctrl+C stop

    # Epoch counter and loss-history lists are shared across ALL starts,
    # so the final convergence plot shows one continuous timeline rather
    # than restarting from zero at each start.
    global_epoch = 0
    continuous_epochs = []
    continuous_train_infid = []
    continuous_test_infid = []

    # The mesh/model are created once and reused across starts; each
    # start just reinitializes their parameters in-place (see below).
    processor = MeshClass(n_modes=n)
    calib_model = CalibrationModel(processor)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(calib_model.parameters(), lr=1e-7)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

    for start_idx in range(num_starts):
        print(f" START {start_idx + 1}/{num_starts} ".center(75, '='))

        if global_epoch >= max_global_epochs:
            print(f"Max global epochs ({max_global_epochs}) reached. Terminating optimization.")
            break

        reinitialize_processor_parameters(processor, current_2pi)

        print("Running automated LR finder...")
        optimal_lr = lr_finder(
            calib_model,
            optimizer,
            criterion,
            I_train,
            U_mag_train,
            plot_result=(plot_lr_finder or start_idx == 0),
        )
        optimizer.param_groups[0]["lr"] = optimal_lr
        optimizer.state.clear()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

        best_loss_this_start = float('inf')
        epochs_no_improve = 0
        escape_attempts = 0
        final_test_fid = 0.0
        start_failed = True  # optimistic default flipped to False only on target reached

        print(f"Starting training (Target Fidelity: {target_fidelity})...")
        start_time = time.perf_counter()

        for global_epoch in range(global_epoch, max_global_epochs):

            calib_model.train()
            optimizer.zero_grad(set_to_none=False)
            U_mag_pred_train = calib_model(I_train)
            loss_train = criterion(U_mag_pred_train, U_mag_train)
            loss_train.backward()
            optimizer.step()
            # apply_xtalk_bounds(calib_model.mesh)
            scheduler.step(loss_train.detach())

            # --- STAGNATION TRACKING ---
            current_loss = loss_train.item()

            # Absolute improvement of the training MSE.
            if current_loss < best_loss_this_start - min_loss_delta:
                best_loss_this_start = current_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # --- PERIODIC TEST-SET EVALUATION ---
            if global_epoch % check_interval == 0:
                elapsed_time = time.perf_counter() - start_time
                start_time = time.perf_counter()

                calib_model.eval()
                with torch.inference_mode():
                    U_mag_pred_test = calib_model(I_test)

                    mean_fid_train = amplitude_fidelity(U_train, U_mag_pred_train.detach()).mean().item()
                    mean_fid_test = amplitude_fidelity(U_test, U_mag_pred_test).mean().item()

                print(
                    f"Global Epoch {global_epoch:6d} |",
                    f"LR: {optimizer.param_groups[0]['lr']:.2e} |",
                    f"Train Fid: {mean_fid_train:.5f} |",
                    f"Test Fid: {mean_fid_test:.5f} |",
                    f"Loss: {loss_train.item():.2e} |",
                    f"Elapsed time: {elapsed_time:.2f} s",
                )

                continuous_epochs.append(global_epoch)
                continuous_train_infid.append(1.0 - mean_fid_train)
                continuous_test_infid.append(1.0 - mean_fid_test)

                if mean_fid_test > best_overall_fidelity:
                    best_overall_fidelity = mean_fid_test
                    best_model_weights = {k: v.clone() for k, v in calib_model.state_dict().items()}

                if mean_fid_test >= target_fidelity:
                    print(f"Target testing fidelity reached at Global Epoch {global_epoch}!")
                    start_failed = False
                    break

            # --- TARGETED CROSS-TALK ESCAPE ---
            if epochs_no_improve >= escape_patience:
                if escape_attempts >= max_escape_attempts:
                    print(f"\nMax escape attempts ({max_escape_attempts}) reached. Triggering next start...")
                    break
                else:
                    escape_attempts, epochs_no_improve, trigger_next_start, scheduler = targeted_xtalk_escape(
                        calib_model=calib_model,
                        optimizer=optimizer,
                        optimal_lr=optimal_lr,
                        scheduler=scheduler,
                        escape_attempts=escape_attempts,
                        max_escape_attempts=max_escape_attempts,
                        escape_patience=escape_attempts,
                        epochs_no_improve=epochs_no_improve,
                    )

        # --- END OF RUN DIAGNOSTICS ---
        run_start_diagnostics(
            start_idx=start_idx,
            start_failed=start_failed,
            final_test_fid=final_test_fid,
            calib_model=calib_model,
            circular_abs_error=circular_abs_error,
            gt_processor=gt_processor,
        )

        # --- TERMINATION CHECK ---
        if interrupted or best_overall_fidelity >= target_fidelity or global_epoch >= max_global_epochs:
            if best_overall_fidelity >= target_fidelity:
                print("\nGlobal target fidelity achieved! Terminating multi-start search early.")
            break

    # FINAL EVALUATION & VISUALIZATIONS
    if interrupted:
        print("\nSearch interrupted by user (Ctrl+C).")

    if best_model_weights is not None:
        print(f"Loading best model with Test Fidelity: {best_overall_fidelity:.5f}")
        calib_model.load_state_dict(best_model_weights)
    else:
        # Interrupted before any check_interval evaluation ever ran (e.g.
        # within the first `check_interval` epochs of the very first
        # start): there is no tracked "best" checkpoint yet, so fall back
        # to whatever the model currently holds rather than crashing on
        # `load_state_dict(None)`.
        print(
            "No checkpoint was recorded yet (interrupted very early) -- "
            "evaluating the current, largely untrained model instead."
        )

    calib_model.eval()

    with torch.no_grad():
        U_mag_pred_final_test = calib_model(I_test)
        final_test_fidelities = amplitude_fidelity(U_test, U_mag_pred_final_test).numpy()

    # Full training history across ALL starts on one timeline.
    plt.figure(figsize=(9, 5))
    plt.semilogy(
        continuous_epochs,
        continuous_train_infid,
        color='blue',
        label='Train Set',
        marker='o',
        markersize=3,
        linestyle='-',
    )
    plt.semilogy(
        continuous_epochs,
        continuous_test_infid,
        color='orange',
        label='Test Set',
        marker='x',
        markersize=3,
        linestyle='--',
    )
    plt.xlabel('Global Epoch', fontsize=12)
    plt.ylabel('Mean Infidelity (1 - F)', fontsize=12)
    plt.title('Continuous Shotgun Optimization (Train vs Test)', fontsize=14)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Distribution of per-sample fidelities for the best model found.
    plt.figure(figsize=(8, 5))
    plt.hist(final_test_fidelities, bins=30, color='green', alpha=0.7, edgecolor='black')
    plt.xlabel('Amplitude Fidelity', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Best Test Fidelities Distribution (Mean: {np.mean(final_test_fidelities):.4f})', fontsize=14)
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
