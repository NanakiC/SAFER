# SAFER Review-Stage Interface Scaffold

This archive is an anonymized review-stage interface scaffold. It exposes the integration boundary, module organization, configuration surface, tensor conventions, and high-level data flow used to attach SAFER to the forecasting backbone. It is not the complete research implementation.

## SFCE disclosure boundary

Disclosed:

- The future-proxy estimator interface.
- The linear inertia projection used to construct the initial future proxy.
- The public shift-related configuration surface and tensor conventions.

Withheld:

- The conditional transition-density parameterization.
- The shift score and probability calibration.
- The inertia-preserving and shift-sensitive decomposition.
- The reliability filtering rule and SFCE auxiliary-loss implementation.

## ICC disclosure boundary

Disclosed:

- The instantaneous-correction interface.
- The high-level residual form in which a learned correction mask modulates the safe future proxy.
- The correction scaling parameter and auxiliary-loss boundary.

Withheld:

- The instantaneous relation-graph construction and sparsification rule.
- The counterfactual perturbation and intervention procedure.
- The correction-mask network, calibration target, and counterfactual loss.
- Internal correction diagnostics.

## DCR disclosure boundary

Disclosed:

- The delayed historical-view construction.
- The latent slot count, state dimension, queries, and projection interfaces.
- The public delayed-loss names and input/output boundary.

Withheld:

- The latent-state pooling computation.
- The delayed transition graph and state-change predictor.
- The intervention-effect target construction.
- The transition, intervention, sparsity, and non-collapse loss computations.

## Other disclosed components

- The plug-and-play wrapper and its backbone integration points.
- The normalized input and output flow.
- The training objective names, public hyperparameters, and experiment scripts.
- The expected tensor conventions and auxiliary-loss interface.

Internal diagnostics and unpublished engineering utilities are not included.

The withheld methods raise `NotImplementedError` with an explicit review-scaffold message. Consequently, SAFER-enabled training and evaluation are intentionally unavailable in this archive. Backbone-only execution remains represented by the original experiment structure. A complete anonymized implementation is intended for release after the review process.

## Main interfaces

`SAFERPlugin.apply_causal_correction(dec_out, x_enc_norm, y, is_training)` accepts batch-first tensors and returns a calibrated forecast together with auxiliary losses in the complete implementation.

`DelayedCausalRegularizer.previous_view(x_enc_norm)` constructs the lagged historical view disclosed in the paper.

`DelayedCausalRegularizer.forward(previous_hidden, current_hidden)` returns delayed transition, intervention, sparsity, and variance losses in the complete implementation.

