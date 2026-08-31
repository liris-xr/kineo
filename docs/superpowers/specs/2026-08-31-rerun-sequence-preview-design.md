# Rerun sequence preview — design

**Date:** 2026-08-31
**Status:** approved, not implemented

## Problem

A preprocessed sequence can only be inspected today by running the whole
pipeline with `RerunExportStage` enabled, or, for EgoHumans alone, by
`scripts/viz_egohumans.py`, which reads the *raw* dataset through aitviewer
and cv2 rather than the preprocessed annotations.

Neither answers the question the ground truth actually raises: do the
annotations this repository wrote line up with the videos they annotate?
That needs a per-dataset entry point that loads a preprocessed sequence and
shows its ground truth — 3D skeletons and cameras in one view, 2D keypoints
and boxes over the footage in another.

## Scope

In: AIST++, Human3.6M, EgoHumans — the three datasets with visual ground
truth.

Out: CHiME-6 (audio only, no keypoints or cameras), `panoptic/` and
`egoexo4d/` (no code in the repository), the aitviewer modules
`kineo/visualization/viz_2d.py` and `viz_3d.py`, and
`kineo/datasets/egohumans/egohumans_viz.py`, which keeps its own purpose of
inspecting the raw dataset before preprocessing.

## Architecture

Three layers, each usable without the one above it.

```
scripts/preview_{aistpp,h36m,egohumans}_sequence.py   argparse, build dataset
    -> kineo/visualization/sequence_preview.py        compose one sequence
        -> kineo/visualization/viz_rerun.py           log one data type
```

### `kineo/visualization/viz_rerun.py` (new)

Rerun logging primitives. Each function logs exactly one kind of structured
data and holds no state.

Moved verbatim out of `kineo/pipeline/stages/rerun_export.py`, which then
imports them from here:

- `log_cameras`
- `log_keypoints_2d`, `log_keypoints_3d`
- `log_skeletons_2d`, `log_skeletons_3d`
- `log_videos`, `quality_to_crf`

They are moved unchanged. They already log one kind of data each, and
rewriting them would put the pipeline's existing rerun output at risk for no
gain. SMPL and world-reconstruction logging stay in the stage: both are
pipeline outputs rather than annotations, and `log_smpl` pulls in `smplx`.

New, each split into a pure conversion and a thin logging wrapper so the
conversion is testable without a recording stream:

- `log_bboxes_2d` — `rr.Boxes2D` per view and subject. No equivalent exists
  today.
- `log_video_asset` — `rr.AssetVideo` plus one `rr.VideoFrameReference` per
  timeline step. The file on disk is referenced, not re-encoded, which is
  what separates this from `log_videos`.
- `log_image_frames` — `rr.EncodedImage` referencing each JPEG on disk, for
  views backed by an image sequence.

Entity paths follow the convention the stage already uses, so a preview and
a pipeline export can be read side by side:

| Entity | Content |
| --- | --- |
| `ground_truth/cameras/{view_id}` | `rr.Pinhole` and `rr.Transform3D` |
| `ground_truth/cameras/{view_id}/rgb` | video asset or image frames |
| `ground_truth/cameras/{view_id}/keypoints_2d_{subject_id}` | 2D keypoints |
| `ground_truth/cameras/{view_id}/bboxes_2d_{subject_id}` | 2D boxes |
| `ground_truth/skeletons_3d_{joints,bones}_{subject_id}` | 3D skeleton |

Media and 2D overlays share the pinhole entity, so the overlays project onto
the footage.

### `kineo/visualization/sequence_preview.py` (new)

One composition function:

```python
def preview_sequence(
    sequence: KeypointsSequence,
    fps: float,
    output_path: str | None = None,
) -> None:
```

It logs the ground-truth cameras and 3D skeletons, then, per view, the media
followed by that view's 2D keypoints and boxes. A view's media kind is read
off its frame loader: `VideoLoader` takes the `rr.AssetVideo` path,
`ImagesLoader` the encoded-image path. `output_path` of `None` spawns the
viewer; otherwise the recording is written to that `.rrd`.

**Timeline.** Steps are indices into the sequence's
`global_time_reference` annotation when it carries one, and plain frame
indices otherwise. This is what keeps AIST++ *raw* honest: its 3D keypoints
are numbered from the start of the annotated window while its 2D keypoints
and video frames are numbered from the start of each camera's own recording,
so a shared timeline that ignores the reference shows 3D and 2D drifting
apart by hundreds of frames, differently per view. Human3.6M and EgoHumans
are frame-aligned, and the identity mapping is correct for them.

### `kineo/datasets/egohumans/egohumans_dataset.py` (new)

`EgoHumansSequenceDataset`, shaped like `H36MSequenceDataset`: it reads
`egohumans_sequences.json`, builds an `ImagesLoader` per view from the view's
`images_dir` and `fps`, and loads the annotations through
`annotations_io.load_sequence_annotations`.

EgoHumans is the only dataset in the repository whose sequences have no
dataset class, which is why `experiments/egohumans_eval.py` opens and
`from_dict`s five annotation files by hand. That block is replaced by
iterating this dataset; the script keeps its own filtering of non-static
views, which is an evaluation decision rather than a loading one.

### `scripts/preview_{aistpp,h36m,egohumans}_sequence.py` (new)

One script per dataset, each roughly 25 lines: argparse over the dataset
path, `--sequence` to pick a sequence by name, `--save` to write an `.rrd`
instead of spawning the viewer; build the dataset; call `preview_sequence`.
AIST++ additionally takes `--variant` (`raw` or `refined`), which its
sequence listings are already split by.

## Testing

- `EgoHumansSequenceDataset` against a fixture directory holding a small
  `egohumans_sequences.json`, a couple of JPEGs per view, and annotation
  files: the views load, the frame timestamps follow the declared fps, and
  the annotations arrive keyed by kind.
- The pure conversions behind `log_bboxes_2d`, `log_video_asset` and
  `log_image_frames`, on tensors, with no recording stream open.
- The moved functions keep their existing behaviour; the import site in
  `rerun_export.py` is the only thing that changes, and a benchmark config
  that enables the stage still exports.

The logging wrappers are not unit tested. Verification is running each of
the three preview scripts against the datasets on disk and confirming the
skeletons sit on the subjects and the boxes on the people.

## Risks

- `log_videos` and `log_video_asset` are two ways to put footage in a
  recording. They stay separate on purpose: the stage's re-encode handles
  resampled pipeline frames, and the asset path handles untouched files. A
  later change may fold one into the other; this design does not.
- `rr.AssetVideo` leaves the file unread until the viewer opens it, so a
  moved or deleted video surfaces as an empty view rather than an error at
  logging time.
