#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:57:51 2026

@author: Francesco Ceccarelli

Multi-start calibration of a UPP mesh (Bell/Clements) via gradient descent
on the amplitude-only forward model, with a targeted "cross-talk escape"
mechanism to recover from stagnation and curriculum learning for the 
non-linear thermal phase components.

Design summary
---------------
1. MULTI-START: each start reinitializes the mesh with random static
   phases and re-runs the automatic LR finder.
2. STAGNATION DETECTION: per-epoch, based on absolute improvement of raw MSE.
3. TARGETED ESCAPE: specific resets on anomalous cross-talk matrix entries.
4. CURRICULUM LEARNING (NON-LINEAR): The non-linear thermal factor (nl_factor)
   is initially frozen to 0 to allow the optimizer to find the correct general
   basin. Once test fidelity hits 0.99, it is unfrozen and jointly optimized 
   to reach the final precision target.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib.pyplot as plt
import traceback

from calippo.model import Bell, Clements
from calippo.tools import amplitude_fidelity


class CalibrationModel(nn.Module):
    """
    Wrapper to train the mesh parameters while strictly enforcing
    the physical constraints (masks) during the forward pass.
    """
    def __init__(self, mesh_instance):
        super().__init__()
        self.mesh = mesh_instance

    def forward(self, currents):
        # 1. Apply masks so gradients only flow through physically-meaningful
        #    parameters (the optimizer must not "invent" nonexistent coupling
        #    paths or heaters).
        sp_active = self.mesh.static_phases * self.mesh.static_phases_mask
        xt_active = self.mesh.xtalk * self.mesh.xtalk_mask

        # 2. Phase model: static phase + current-dependent cross-talk
        #    contribution with optional non-linear conditioning.
        I2 = currents ** 2
        
        if self.mesh.nonlinear:
            # Active non-linear factors
            nl_active = self.mesh.nl_factor * self.mesh.static_phases_mask
            I2_eff = I2 * (1 + nl_active * I2)
        else:
            I2_eff = I2
            
        phases = sp_active + I2_eff @ xt_active.T

        # 3. Native unitary propagation.
        U = self.mesh.U_from_phases(phases)
        return torch.abs(U)


def lr_finder(model, optimizer, criterion, I_train, U_target_train,
              init_lr=1e-7, max_lr=10.0, steps=100, plot_result=False):
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
        plt.scatter(optimal_lr, losses[steepest_idx], color='red', s=50, zorder=5,
                    label=f'Optimal LR: {optimal_lr:.2e}')
        plt.scatter(lrs[first_min_idx], losses[first_min_idx], color='orange', s=50, zorder=4,
                    label='First Minimum (Noise Start)', marker='X')

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


if __name__ == "__main__":

    calib_model = None
    gt_processor = None

    try:
        # ==========================================
        # CONFIGURATION
        # ==========================================
        mesh_choice = "Clements"
        MeshClass = Bell if mesh_choice == "Bell" else Clements

        n = 6
        train_ratio = 0.8
        current_2pi = 16

        load_external_data = True
        data_filepath = "../../data/unitary_measurement_20260121_1.npy"

        # --- Multi-start configuration ---
        num_starts = 10
        target_fidelity = 0.9995
        check_interval = 100
        max_global_epochs = 1000000

        # --- Stagnation & targeted escape configuration ---
        escape_patience = 200
        max_escape_attempts = 5
        min_loss_delta = 1e-6

        plot_lr_finder_every_start = True

        # ==========================================
        # 1. DATA GENERATION / LOADING
        # ==========================================
        if load_external_data:
            print(f"Loading external experimental data from {data_filepath}...")
            with open(data_filepath, 'rb') as f:
                u_th = np.load(f)
                U_np_reshaped = np.load(f)
                currents = np.load(f)

            loaded_data = np.load(data_filepath, allow_pickle=True)
            if isinstance(loaded_data, np.ndarray) and loaded_data.size == 1:
                loaded_data = loaded_data.item()

            from calippo.tools import sinkhorn
            
            I_exp = torch.tensor(currents, dtype=torch.float32)
            U_exp = torch.tensor(U_np_reshaped, dtype=torch.complex128)
            U_exp = torch.sqrt(sinkhorn(U_exp)[0])
            U_mag_exp = torch.abs(U_exp)

            total_samples = I_exp.shape[0]
            n_train = int(total_samples * train_ratio)
            n_test = total_samples - n_train
        else:
            print("Generating mock experimental data...")
            total_samples = 10000
            n_train = int(total_samples * train_ratio)
            n_test = total_samples - n_train

            # Generiamo GT con non-linearità abilitata per testare l'efficacia del learning
            gt_processor = MeshClass(n_modes=n, nonlinear=True)
            I_exp = gt_processor.generate_random_currents(current_ampl=current_2pi, batch_size=total_samples)
            with torch.no_grad():
                phases_exp = gt_processor.phases_from_currents(I_exp)
                U_exp = gt_processor.U_from_phases(phases_exp)
                U_mag_exp = torch.abs(U_exp)

        print(f"Total Samples: {total_samples} (Train: {n_train}, Test: {n_test})")

        I_train, I_test = I_exp[:n_train], I_exp[n_train:]
        U_train, U_test = U_exp[:n_train], U_exp[n_train:]
        U_mag_train, U_mag_test = U_mag_exp[:n_train], U_mag_exp[n_train:]

        # ==========================================
        # MULTI-START OPTIMIZATION LOOP
        # ==========================================
        best_overall_fidelity = -1.0
        best_model_weights = None
        interrupted = False

        global_epoch = 0
        continuous_epochs = []
        continuous_train_infid = []
        continuous_test_infid = []

        # Mesh inizializzata CON modello non-lineare attivo
        processor = MeshClass(n_modes=n, nonlinear=True)
        calib_model = CalibrationModel(processor)

        criterion = nn.MSELoss()

        for start_idx in range(num_starts):
            if global_epoch >= max_global_epochs:
                break

            print(f" START {start_idx + 1}/{num_starts} ".center(75, '='))

            nl_unfrozen = False # Flag per il curriculum learning

            # --- Random reinitialization for this start ---
            with torch.no_grad():
                random_phases = torch.rand(processor.n_phases, dtype=processor.static_phases.dtype) * 2 * torch.pi
                processor.static_phases.copy_(random_phases)

                processor.xtalk.copy_(torch.eye(processor.n_phases, dtype=processor.xtalk.dtype))
                processor.xtalk.mul_(2 * torch.pi / current_2pi**2)

                processor.static_phases.mul_(processor.static_phases_mask)
                processor.xtalk.mul_(processor.xtalk_mask)

                # Resetta e congela il fattore non-lineare all'inizio
                if processor.nonlinear:
                    processor.nl_factor.zero_()
                    processor.nl_factor.requires_grad = False

            # L'ottimizzatore prende tutti i parametri, ma ignorerà nl_factor (perché requires_grad=False)
            optimizer = optim.Adam(calib_model.parameters(), lr=1e-7)

            print("Running automated LR finder...")
            try:
                optimal_lr = lr_finder(
                    calib_model, optimizer, criterion, I_train, U_mag_train,
                    plot_result=(plot_lr_finder_every_start or start_idx == 0),
                )
            except KeyboardInterrupt:
                # Ctrl+C during the LR sweep: nothing meaningful has been
                # trained yet for this start, so just stop the whole search
                # and fall through to the final evaluation with whatever the
                # best previous start achieved (if any).
                print("\n[INTERRUPTED] Ctrl+C received during LR finder. Stopping the search.")
                interrupted = True
                break
            
            optimizer.param_groups[0]['lr'] = optimal_lr
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

            best_loss_this_start = float('inf')
            epochs_no_improve = 0
            escape_attempts = 0
            final_test_fid = 0.0
            start_failed = True

            print(f"Starting training (Target Fidelity: {target_fidelity})...")

            while True:
                global_epoch += 1

                if global_epoch > max_global_epochs:
                    print(f"\nMax global epochs ({max_global_epochs}) reached! Terminating optimization.")
                    break

                # --- TRAIN STEP ---
                # This is where the process spends nearly all of its time, so
                # it's the most likely place to catch a Ctrl+C. On interrupt:
                # stop training for this start cleanly (keep whatever was
                # already learned), run its diagnostics as usual, then let
                # the surrounding loops see `interrupted` and unwind straight
                # to the final evaluation/plots instead of crashing.
                try:
                    calib_model.train()
                    optimizer.zero_grad()

                    U_mag_pred_train = calib_model(I_train)
                    loss_train = criterion(U_mag_pred_train, U_mag_train)

                    loss_train.backward()
                    optimizer.step()
                    scheduler.step(loss_train.detach())
                except KeyboardInterrupt:
                    print(f"\n[INTERRUPTED] Ctrl+C received at Global Epoch {global_epoch}. "
                          f"Stopping training and moving to final evaluation.")
                    interrupted = True
                    break

                # --- STAGNATION TRACKING ---
                # Absolute (not relative) improvement of the raw training MSE,
                # checked every epoch. Using an absolute threshold avoids the
                # failure mode where a shrinking LR (via the scheduler) makes
                # the *relative* improvement look small even on a run that is
                # still converging normally.
                if loss_train.item() < best_loss_this_start - min_loss_delta:
                    best_loss_this_start = loss_train.item()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # --- PERIODIC TEST-SET EVALUATION ---
                if global_epoch % check_interval == 0:
                    calib_model.eval()
                    with torch.no_grad():
                        U_mag_pred_test = calib_model(I_test)
                        mean_fid_train = amplitude_fidelity(U_train, U_mag_pred_train).mean().item()
                        mean_fid_test = amplitude_fidelity(U_test, U_mag_pred_test).mean().item()

                    current_lr = optimizer.param_groups[0]['lr']
                    infidelity_train = 1.0 - mean_fid_train
                    infidelity_test = 1.0 - mean_fid_test

                    continuous_epochs.append(global_epoch)
                    continuous_train_infid.append(infidelity_train)
                    continuous_test_infid.append(infidelity_test)

                    print(f"Global Epoch {global_epoch:6d} | LR: {current_lr:.2e} | "
                          f"Train Fid: {mean_fid_train:.5f} | Test Fid: {mean_fid_test:.5f} | Loss: {loss_train.item():.2e}")

                    final_test_fid = mean_fid_test

                    if final_test_fid > best_overall_fidelity:
                        best_overall_fidelity = final_test_fid
                        best_model_weights = copy.deepcopy(calib_model.state_dict())

                    if mean_fid_test >= target_fidelity:
                        print(f"\nTarget testing fidelity reached at Global Epoch {global_epoch}!")
                        start_failed = False
                        break

                    # ----------------------------------------------------------
                    # CURRICULUM LEARNING: SCONGELAMENTO FATTORE NON-LINEARE
                    # ----------------------------------------------------------
                    if processor.nonlinear and mean_fid_test > 0.99 and not nl_unfrozen:
                        print(f"\n[PHASE 2] Test Fidelity > 0.99 reached! Unfreezing 'nl_factor' for non-linear optimization.")
                        processor.nl_factor.requires_grad = True
                        nl_unfrozen = True

                # ----------------------------------------------------------
                # TARGETED CROSS-TALK ESCAPE
                # ----------------------------------------------------------
                # Rather than perturbing every parameter blindly, inspect the
                # cross-talk matrix for entries that are statistical outliers
                # relative to the fitted diagonal (heater self-efficiency),
                # and reset only those -- plus the static phase tied to any
                # anomalous diagonal entry, since a bad heater-efficiency
                # estimate is often compensated by an equally-wrong static
                # phase, and the two need to be broken out of that joint
                # local minimum together.
                # if epochs_no_improve >= escape_patience:
                #     if escape_attempts < max_escape_attempts:
                #         with torch.no_grad():
                #             mesh = calib_model.mesh
                #             xt_active = mesh.xtalk * mesh.xtalk_mask
                #             n_p = xt_active.shape[0]

                #             diag_idx = torch.arange(n_p, device=mesh.xtalk.device)
                #             diag = xt_active[diag_idx, diag_idx]
                #             diag_mask = mesh.xtalk_mask[diag_idx, diag_idx].bool()

                #             valid_diag = diag[diag_mask]
                #             if len(valid_diag) > 0:
                #                 mean_diag = valid_diag.abs().mean().clamp_min(1e-8)
                #             else:
                #                 mean_diag = torch.tensor(1e-8, device=diag.device)

                #             diag_deviation = (diag.abs() - mean_diag).abs() / mean_diag
                #             bad_diag_mask = (diag_deviation > 0.50) & diag_mask

                #             offdiag_mask = (mesh.xtalk_mask.bool() & (~torch.eye(n_p, dtype=torch.bool, device=mesh.xtalk.device)))
                #             offdiag_ratio_2d = mesh.xtalk.abs() / mean_diag
                #             bad_offdiag_mask_2d = offdiag_mask & (offdiag_ratio_2d > 0.30)

                #             has_bad_diag = bad_diag_mask.any()
                #             has_bad_offdiag = bad_offdiag_mask_2d.any()

                #             if has_bad_diag or has_bad_offdiag:
                #                 escape_attempts += 1

                #                 if has_bad_offdiag:
                #                     target_offdiag = 0.10 * mean_diag
                #                     mesh.xtalk.data[bad_offdiag_mask_2d] = torch.sign(mesh.xtalk.data[bad_offdiag_mask_2d]) * target_offdiag

                #                 if has_bad_diag:
                #                     bad_diag_indices = diag_idx[bad_diag_mask]
                #                     target_diag = mean_diag

                #                     mesh.xtalk.data[bad_diag_indices, bad_diag_indices] = torch.sign(mesh.xtalk.data[bad_diag_indices, bad_diag_indices]) * target_diag

                #                     bad_sp_mask = bad_diag_mask & mesh.static_phases_mask.bool()
                #                     bad_sp_indices = diag_idx[bad_sp_mask]

                #                     if len(bad_sp_indices) > 0:
                #                         random_phases = torch.rand(len(bad_sp_indices), dtype=mesh.static_phases.dtype, device=mesh.static_phases.device) * 2 * torch.pi
                #                         mesh.static_phases.data[bad_sp_indices] = random_phases

                #                 optimizer.state.clear()
                #                 epochs_no_improve = 0
                #                 optimizer.param_groups[0]['lr'] = optimal_lr
                #                 scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100)

                #                 num_bad_diag = bad_diag_mask.sum().item()
                #                 num_bad_offdiag = bad_offdiag_mask_2d.sum().item()

                #                 print(f"\n[TARGETED XTALK ESCAPE] Reset {num_bad_diag} diagonal (with static phases) and {num_bad_offdiag} off-diagonal "
                #                       f"anomalous active coefficients (Attempt {escape_attempts}/{max_escape_attempts}).")
                #             else:
                #                 print(f"\nLoss stagnated for {escape_patience} epochs, but no anomalous xtalk found. Triggering next start...")
                #                 break
                #     else:
                #         print(f"\nMax escape attempts ({max_escape_attempts}) reached. Triggering next start...")
                #         break

            # ==========================================
            # PER-START DIAGNOSTICS
            # ==========================================
            # Printed/plotted for EVERY start, successful or not, since a
            # failed start's parameter map is often the most informative one
            # for spotting a systematic issue with the escape heuristics.
            status_str = "failed to reach target" if start_failed else "successfully reached target"
            print(f"\n[DIAGNOSTICS] Start {start_idx + 1} {status_str} (Final Test Fid: {final_test_fid:.5f}).")

            with torch.no_grad():
                est_sp = np.mod((calib_model.mesh.static_phases * calib_model.mesh.static_phases_mask).cpu().numpy(), 2 * np.pi)
                est_xt = (calib_model.mesh.xtalk * calib_model.mesh.xtalk_mask).cpu().numpy()
                if calib_model.mesh.nonlinear:
                    est_nl = (calib_model.mesh.nl_factor * calib_model.mesh.static_phases_mask).cpu().numpy()

            if gt_processor is not None:
                # Mock data: ground truth is available, so report fit errors.
                print("Analyzing parameter errors against Ground Truth...")
                with torch.no_grad():
                    gt_sp = np.mod((gt_processor.static_phases * gt_processor.static_phases_mask).cpu().numpy(), 2 * np.pi)
                    gt_xt = (gt_processor.xtalk * gt_processor.xtalk_mask).cpu().numpy()
                    if gt_processor.nonlinear:
                        gt_nl = (gt_processor.nl_factor * gt_processor.static_phases_mask).cpu().numpy()

                # Static phases are circular quantities: use the wrap-around
                # aware distance so that e.g. 0.01 rad vs 6.27 rad is
                # correctly reported as a ~0.02 rad error, not a ~6.26 rad
                # error (see `circular_abs_error` docstring).
                sp_error = circular_abs_error(est_sp, gt_sp)
                xt_error = np.abs(est_xt - gt_xt)

                print(f"  -> Static Phases [0, 2pi] - Max Circular Error: {np.max(sp_error):.4e} | Mean Error: {np.mean(sp_error[gt_processor.static_phases_mask.cpu().numpy()]):.4e}")
                print(f"  -> Cross-Talk Matrix - Max Absolute Error: {np.max(xt_error):.4e} | Mean Error: {np.mean(xt_error[gt_processor.xtalk_mask.cpu().numpy() > 0]):.4e}")

                if gt_processor.nonlinear:
                    nl_error = np.abs(est_nl - gt_nl)
                    print(f"  -> Non-Linear Factor - Max Absolute Error: {np.max(nl_error):.4e} | Mean Error: {np.mean(nl_error[gt_processor.static_phases_mask.cpu().numpy()]):.4e}")

                worst_sp_idx = np.argmax(sp_error)
                print(f"  -> Worst Static Phase index: {worst_sp_idx} (Est: {est_sp[worst_sp_idx]:.4f}, GT: {gt_sp[worst_sp_idx]:.4f})")

                worst_xt_idx = np.unravel_index(np.argmax(xt_error), xt_error.shape)
                print(f"  -> Worst Cross-Talk coordinates: {worst_xt_idx} (Est: {est_xt[worst_xt_idx]:.4f}, GT: {gt_xt[worst_xt_idx]:.4f})")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                im0 = axes[0].imshow(est_xt - gt_xt, cmap='coolwarm', aspect='auto')
                axes[0].set_title(f"Start {start_idx + 1}: Cross-Talk Error (Est - GT)")
                fig.colorbar(im0, ax=axes[0])

                axes[1].plot(gt_sp, label='GT Static Phases', marker='o', alpha=0.7)
                axes[1].plot(est_sp, label='Est Static Phases [0, 2pi]', marker='x', alpha=0.7)
                axes[1].set_title(f"Start {start_idx + 1}: Static Phases Comparison")
                axes[1].legend()
                axes[1].grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                plt.show()

            else:
                # Real experimental data: no ground truth, just show the estimates.
                print("Plotting estimated parameters (No Ground Truth available for experimental data)...")

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                im0 = axes[0].imshow(est_xt, cmap='viridis', aspect='auto')
                axes[0].set_title(f"Start {start_idx + 1}: Estimated Cross-Talk Matrix")
                fig.colorbar(im0, ax=axes[0])

                axes[1].plot(est_sp, label='Est Static Phases [0, 2pi]', marker='x', color='orange')
                axes[1].set_title(f"Start {start_idx + 1}: Estimated Static Phases")
                axes[1].legend()
                axes[1].grid(True, linestyle=":", alpha=0.6)
                plt.tight_layout()
                plt.show()

            if interrupted or best_overall_fidelity >= target_fidelity or global_epoch >= max_global_epochs:
                if best_overall_fidelity >= target_fidelity:
                    print("\nGlobal target fidelity achieved! Terminating multi-start search early.")
                break

        # ==========================================
        # FINAL EVALUATION & VISUALIZATIONS
        # ==========================================
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
            print("No checkpoint was recorded yet (interrupted very early) -- "
                  "evaluating the current, largely untrained model instead.")
        calib_model.eval()

        with torch.no_grad():
            U_mag_pred_final_test = calib_model(I_test)
            final_test_fidelities = amplitude_fidelity(U_test, U_mag_pred_final_test).numpy()

        # Plot 1: full training history across ALL starts on one timeline.
        plt.figure(figsize=(9, 5))
        plt.semilogy(continuous_epochs, continuous_train_infid, color='blue', label='Train Set', marker='o', markersize=3, linestyle='-')
        plt.semilogy(continuous_epochs, continuous_test_infid, color='orange', label='Test Set', marker='x', markersize=3, linestyle='--')
        plt.xlabel('Global Epoch', fontsize=12)
        plt.ylabel('Mean Infidelity (1 - F)', fontsize=12)
        plt.title('Continuous Shotgun Optimization (Train vs Test)', fontsize=14)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot 2: distribution of per-sample fidelities for the best model found.
        plt.figure(figsize=(8, 5))
        plt.hist(final_test_fidelities, bins=30, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Amplitude Fidelity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Best Test Fidelities Distribution (Mean: {np.mean(final_test_fidelities):.4f})', fontsize=14)
        plt.grid(True, ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    except KeyboardInterrupt:
        # Safety net: catches a Ctrl+C that slips through somewhere not
        # explicitly wrapped above (e.g. while a blocking plt.show() window
        # is open). We may be mid-computation with no clean "best model" to
        # fall back on here, so just exit quietly instead of dumping a stack
        # trace -- whatever was already printed/plotted up to this point is
        # still there for inspection.
        print("\n[INTERRUPTED] Ctrl+C received. Exiting cleanly without further evaluation.")

    except Exception as e:
        traceback.print_exc()

    finally:
        if calib_model is not None:
            pass