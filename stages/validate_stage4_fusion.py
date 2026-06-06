"""Stage 4 Validation — V2V Cooperative Fusion (CARLA).

STATUS: STUB. The previous validation was specific to the retired KITTI
temporal-simulation backend (it auto-detected static GT tracks across a
sequence because temporal V2V only registers static objects correctly). CARLA
provides true simultaneous V2V, so that scoping no longer applies and the
validation must be rebuilt against CARLA ground truth — which is not available
yet.

Intended design once CARLA data exists
---------------------------------------
For each scene (an agent pair A/B at one timestamp), measure the value fusion
adds over single-agent perception, against CARLA GT vehicles in Vehicle A's
frame:
    - A-alone : Vehicle A's Stage 3 boxes, matched to GT.
    - Fused   : Stage 4 fused output, matched to GT.
    Matching: greedy BEV centre distance within class (reuse the per-class
    ``matching.max_dist`` thresholds from config/stage4.yaml).
Report per method:
    - recall_A vs recall_fused (does B cover objects A missed?)
    - precision_A vs precision_fused
    - mean BEV localization error on true positives
    - B-unique true positives (objects only B saw)

Reuse ``stages.stage4_fusion.run_carla`` to produce fused boxes and
``utils.fusion.bev_distance`` for matching. GT comes from
``utils.carla_loader`` (per-agent GT vehicles for the scene).

Thesis metrics to capture once CARLA is wired (schema for the eventual
``validation_results.json`` + MLflow, so the next implementer has a target):
    - n_tp / n_fp / n_fn for each agent SOLO (A-alone, B-alone) and for the
      FUSED output — each matched to CARLA GT in A's frame.
    - recall_improvement = recall_fused - recall_A_alone (the headline V2V gain).
    - b_unique_tp = GT objects detected only by Vehicle B (outside A's view /
      occluded for A) that fusion adds to A's scene.
    - localization_error_A vs localization_error_fused = mean BEV centre error
      on true positives, single-agent vs fused.
    - depth_range_breakdown = the same GT-depth bins as Stage 3
      (0-10m, 10-20m, 20-40m, 40m+): per bin n_pairs, mean depth/centre error.
    - per-class breakdown (Car, Pedestrian) for all of the above counts/errors.
    - mean inference time per frame (seconds), model load excluded.
"""

import sys


def main() -> None:
    """Entry point — not implemented until CARLA validation data is available."""
    # TODO(carla): implement the validation described in the module docstring
    # against a real CARLA export (GT vehicles per scene + Stage 3 / fused boxes).
    raise NotImplementedError(
        "Stage 4 validation is not implemented yet. Rebuild it against CARLA "
        "ground truth once a sample export exists (see this module's docstring)."
    )


if __name__ == "__main__":
    print(__doc__, file=sys.stderr)
    main()
