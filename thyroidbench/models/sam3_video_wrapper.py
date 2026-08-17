"""SAM3 framewise + video adapters for ThyroidBench video experiment.

Bridges the SAM3 Python API
(``sam3.model_builder.build_sam3_video_model`` + the existing
:class:`thyroidbench.models.sam3_wrapper.SAM3Predictor`) to the SAM2-shaped interface
that ``experiments/exp6_cineclip_video/run.py`` expects, so framewise vs video parity
with SAM2 / MedSAM-2 is preserved with minimal model-branching in ``run.py``.

Public API:
    build_sam3_framewise_predictor(device) -> SAM3FramewiseAdapter
    build_sam3_video_predictor(device)     -> SAM3VideoAdapter

**Configuration rationale** (see
``experiments/exp6_cineclip_video/the configuration notes in this module's docstring`` for the full citation
trail). The video adapter follows Meta's official
``examples/sam3_for_sam2_video_task_example.ipynb`` notebook line-for-line:

* ``build_sam3_video_model(apply_temporal_disambiguation=False)`` loads the
  joint detector + tracker checkpoint;
* the tracker (``sam3_model.tracker``) is driven directly; its missing
  backbone attribute is hot-swapped in from
  ``sam3_model.detector.backbone`` so it can compute per-frame visual
  features without invoking the concept detector cascade;
* ``clear_non_cond_mem_around_input`` is overridden to ``False`` post-build
  to match SAM2's default memory-pruning policy at conditioning frames
  (SAM3's ``build_tracker`` hard-codes ``True``);
* the box prompt is supplied in relative ``[0, 1]`` xyxy coords as required
  by ``Sam3TrackerPredictor.add_new_points_or_box``; conversion from
  ``run.py``'s pixel-space xyxy uses ``inference_state['video_width' /
  'video_height']``.

This is the **Promptable Visual Segmentation (PVS)** configuration
described in the SAM3 paper §F.6 — the configuration in which Meta
benchmarks SAM3 head-to-head with SAM2 on the VOS task. It is the
officially supported single-object box-prompt video API, not a workaround.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_CKPT_SAM3 = _ROOT / "pretrained_models" / "sam3_vanilla" / "sam3.pt"


def _resolve_ckpt() -> str:
    if not _CKPT_SAM3.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found at {_CKPT_SAM3}")
    return str(_CKPT_SAM3)


# ── Framewise adapter ──────────────────────────────────────────────────────


class SAM3FramewiseAdapter:
    """SAM2ImagePredictor-shaped adapter around :class:`SAM3Predictor`.

    :class:`SAM3Predictor` returns float logits already up-sampled to the
    original ``(H, W)`` image resolution; this adapter wraps the result in a
    ``(1, H, W)`` array so the existing slice ``low_res_masks[[0]]`` in
    :func:`predict_framewise` (in ``experiments/exp6_cineclip_video/run.py``)
    produces a ``(1, H, W)`` tensor that ``F.interpolate`` then maps identity
    to ``(H, W)`` — preserving symmetry with the SAM2 / MedSAM-2 framewise
    paths.
    """

    def __init__(self, device: str = "cuda"):
        from .sam3_wrapper import SAM3Predictor

        self._inner = SAM3Predictor(device=device)
        self._image: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        self._image = image

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box: np.ndarray | None = None,
        multimask_output: bool = False,
    ):
        if self._image is None:
            raise RuntimeError("set_image must be called before predict")
        if box is None:
            raise ValueError("SAM3FramewiseAdapter requires a box prompt")
        box_arr = np.asarray(box, dtype=np.float32)
        if box_arr.ndim == 2:
            box_arr = box_arr[0]
        logit_hw = self._inner.predict(self._image, box_arr)
        low_res_masks_np = logit_hw[None, ...].astype(np.float32)
        return None, None, low_res_masks_np


def build_sam3_framewise_predictor(device: str = "cuda") -> SAM3FramewiseAdapter:
    return SAM3FramewiseAdapter(device=device)


# ── Video adapter ──────────────────────────────────────────────────────────


class SAM3VideoAdapter:
    """SAM2VideoPredictor-shaped adapter around ``Sam3TrackerPredictor``.

    Implements the PVS (Promptable Visual Segmentation) configuration from
    the SAM3 paper §F.6 by driving the tracker directly with the user's box
    prompt, bypassing the text-conditioned detector. Follows the
    canonical pattern from Meta's
    ``examples/sam3_for_sam2_video_task_example.ipynb`` notebook line for
    line; see ``experiments/exp6_cineclip_video/the configuration notes in this module's docstring`` for the
    citation trail.

    Surface methods match :func:`run_video` in
    ``experiments/exp6_cineclip_video/run.py``::

        init_state(video_path)                -> inference_state
        reset_state(inference_state)          -> clear_all_points_in_video
        add_new_points_or_box(...)            -> (frame_idx, obj_ids, video_res_masks)
        propagate_in_video(inference_state)   -> yields (frame_idx, obj_ids, out_mask_logits)
    """

    def __init__(self, full_video_model: Any):
        self._full = full_video_model
        # Canonical Meta pattern (notebook):
        #     predictor = sam3_model.tracker
        #     predictor.backbone = sam3_model.detector.backbone
        # The tracker is built without a backbone (`build_tracker` defaults
        # ``with_backbone=False``), so :meth:`_get_image_feature` would
        # raise unless ``predictor.backbone`` is set; the joint sam3.pt
        # checkpoint stores the visual backbone under the detector and the
        # notebook hot-swaps that backbone into the tracker so it can
        # compute features per frame without invoking the concept detector
        # heads (transformer, segmentation head, dot-product scoring).
        self._tracker = full_video_model.tracker
        self._tracker.backbone = full_video_model.detector.backbone
        # SAM3 hard-codes ``clear_non_cond_mem_around_input=True`` in
        # ``build_tracker`` (model_builder.py:478); SAM2's default is
        # ``False``. Override for memory-pruning parity at conditioning
        # frames — see the configuration notes in this module's docstring §4.
        self._tracker.clear_non_cond_mem_around_input = False

    def init_state(self, video_path: str) -> dict[str, Any]:
        """Load JPG frames with SAM3-ViT normalization (0.5, 0.5, 0.5) and
        construct the tracker inference state.

        ``Sam3TrackerPredictor.init_state`` (sam3_tracking_predictor.py:54)
        calls ``load_video_frames`` without overriding the ImageNet-default
        normalization in ``sam3.model.utils.sam2_utils.load_video_frames``
        (img_mean=(0.485, 0.456, 0.406), img_std=(0.229, 0.224, 0.225)),
        whereas the SAM3 ViT visual encoder was trained with
        (0.5, 0.5, 0.5)/(0.5, 0.5, 0.5) per ``build_sam3_video_model``
        (model_builder.py:737-738). The official video notebook inherits
        the same defect because it likewise calls
        ``predictor.init_state(video_path=...)`` without overriding. To
        avoid silently feeding the tracker mis-normalized frames, we load
        the frames ourselves with the correct mean/std and then construct
        the tracker state in the explicit-dimensions path
        (sam3_tracking_predictor.py:95-99), which skips the internal
        loader; the frames are then injected into the state dict's
        ``images`` slot under the same key the tracker would have used.
        """
        from sam3.model.utils.sam2_utils import load_video_frames

        device_for_load = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        images, video_height, video_width = load_video_frames(
            video_path=video_path,
            image_size=self._tracker.image_size,
            offload_video_to_cpu=False,
            img_mean=(0.5, 0.5, 0.5),
            img_std=(0.5, 0.5, 0.5),
            compute_device=device_for_load,
        )
        inference_state = self._tracker.init_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=len(images),
        )
        inference_state["images"] = images
        return inference_state

    def reset_state(self, inference_state: dict[str, Any]) -> None:
        # Mirrors SAM2VideoPredictor.reset_state: clear all per-object
        # prompt/output structures so the next prompt starts from a clean
        # slate. ``Sam3TrackerPredictor.clear_all_points_in_video`` is the
        # equivalent in the notebook's "before adding a new prompt" cell.
        self._tracker.clear_all_points_in_video(inference_state)

    def add_new_points_or_box(
        self,
        inference_state: dict[str, Any],
        frame_idx: int,
        obj_id: int,
        box: np.ndarray,
    ):
        H = float(inference_state["video_height"])
        W = float(inference_state["video_width"])
        box_arr = np.asarray(box, dtype=np.float32)
        # Accept (1, 4) or (4,); run.py passes ``box0_j[None]`` (= (1, 4)).
        if box_arr.ndim == 1:
            box_arr = box_arr[None]
        # Pixel-space xyxy → relative xyxy in [0, 1] (the format the
        # SAM3 notebook uses; ``rel_coordinates=True`` is the default).
        rel_box = box_arr.copy()
        rel_box[:, 0::2] /= W
        rel_box[:, 1::2] /= H
        # Clamp tiny numerical overshoot at the image boundary so SAM3's
        # internal coordinate transforms do not see >1.0 values.
        rel_box = np.clip(rel_box, 0.0, 1.0).astype(np.float32)
        # Degenerate-box guard: if jitter pushed the box outside the
        # image and the clamp collapsed it to a line, SAM3 silently
        # produces a useless prompt. Surface the failure loudly so
        # :func:`run_video` can skip the patient like other prompt errors.
        min_w = float((rel_box[:, 2] - rel_box[:, 0]).min())
        min_h = float((rel_box[:, 3] - rel_box[:, 1]).min())
        if min_w < 1e-3 or min_h < 1e-3:
            raise ValueError(
                f"SAM3 box prompt collapsed to a degenerate strip "
                f"after clamping (min_w={min_w:.4e}, min_h={min_h:.4e})."
            )
        frame_idx_out, obj_ids, _low_res_masks, video_res_masks = (
            self._tracker.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=rel_box,
                rel_coordinates=True,
            )
        )
        return frame_idx_out, obj_ids, video_res_masks

    def propagate_in_video(
        self, inference_state: dict[str, Any]
    ) -> Generator[tuple[int, list[int], torch.Tensor], None, None]:
        # SAM3's tracker requires explicit start_frame_idx, max_frame_num
        # _to_track, reverse, and a ``propagate_preflight=True`` flag —
        # exactly the call shape from the official notebook. We surface a
        # SAM2-compatible 3-tuple yield ``(frame_idx, obj_ids,
        # out_mask_logits)`` so the consumer in
        # :func:`run_video` (run.py) needs no model-specific branching.
        for frame_idx, obj_ids, _low_res_masks, video_res_masks, _obj_scores in \
                self._tracker.propagate_in_video(
                    inference_state=inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=None,
                    reverse=False,
                    propagate_preflight=True,
                ):
            yield frame_idx, obj_ids, video_res_masks


def build_sam3_video_predictor(device: str = "cuda") -> SAM3VideoAdapter:
    """Build the SAM3 PVS-mode video adapter (detector cascade bypassed).

    The wrapper follows Meta's ``sam3_for_sam2_video_task_example.ipynb``
    notebook: build the full joint detector + tracker model, then drive
    only the tracker (hot-swapping the detector's backbone in) so SAM3 acts
    as a pure single-object box-prompt tracker comparable to SAM2 and
    MedSAM-2. ``apply_temporal_disambiguation=False`` selects the SAM3
    configuration that zeros the hotstart / re-conditioning /
    masklet-confirmation heuristics and disables memory-selection, leaving
    the tracker's behavior dependent only on the user's frame-0 prompt and
    its memory bank — the closest analog to SAM2's
    ``propagate_in_video``.

    See ``experiments/exp6_cineclip_video/the configuration notes in this module's docstring`` for the full
    citation trail.
    """
    from sam3.model_builder import build_sam3_video_model

    full_model = build_sam3_video_model(
        checkpoint_path=_resolve_ckpt(),
        load_from_HF=False,
        apply_temporal_disambiguation=False,
        device=device,
    )
    return SAM3VideoAdapter(full_model)
