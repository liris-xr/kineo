# Rerun sequence preview — design

**Date:** 2026-08-31
**Status:** implemented

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

New:

- `log_bboxes_2d` — `rr.Boxes2D` per view and subject, straight from the
  annotation's `xyxy`. No equivalent exists today.
- `log_video_asset` — `rr.AssetVideo` plus one `rr.VideoFrameReference` per
  timeline step. Nothing is decoded or re-encoded, which is what separates
  this from `log_videos`, but the encoded file is *carried into the
  recording*: a recording costs what the videos behind it cost. Measured at
  278 MB for one nine-view AIST++ sequence.

Every view goes through `log_video_asset`, image sequences included. Logging
JPEGs as `rr.EncodedImage` was tried and abandoned: it embeds each file, so
one EgoHumans sequence of 4K frames produced a 4.4 GB recording against
759 MB once encoded.

Entity paths follow the convention the stage already uses, so a preview and
a pipeline export can be read side by side:

| Entity | Content |
| --- | --- |
| `ground_truth/cameras/{view_id}` | `rr.Pinhole` and `rr.Transform3D` |
| `ground_truth/cameras/{view_id}/rgb` | video asset |
| `ground_truth/cameras/{view_id}/skeletons_2d_{joints,bones}_{subject_id}` | 2D skeleton |
| `ground_truth/cameras/{view_id}/bboxes_2d_{subject_id}` | 2D boxes |
| `ground_truth/skeletons_3d_{joints,bones}_{subject_id}` | 3D skeleton |

Media and 2D overlays share the pinhole entity, so the overlays project onto
the footage.

### `kineo/visualization/sequence_preview.py` (new)

The composition function, plus the pure re-indexing the timeline needs:

```python
def preview_sequence(
    sequence: KeypointsSequence,
    fps: float,
    output_path: str | None = None,
    max_frames: int | None = None,
) -> None:
```

It logs the ground-truth cameras and 3D skeletons, then, per view, the
footage followed by that view's 2D skeleton and boxes. Joint radii and bone
thicknesses are read in pixels by a 2D view, so they are scaled to the view's
height: the logging defaults are hundredths of a pixel, invisible on a 4K
frame. `output_path` of
`None` spawns the viewer; otherwise the recording is written to that `.rrd`.
`max_frames` shortens the timeline and the annotations on it — not the
footage, which is embedded whole.

**Layout.** `sequence_blueprint` sends a blueprint alongside the recording:
the 3D scene on the left holding the skeletons and the camera frustums, and a
grid on the right holding one 2D view per camera, each with that camera's
footage and the annotations over it. Rerun's automatic layout gives every
entity a view of its own, which for a nine-camera sequence buries the footage
under a view per skeleton. The footage is excluded from the 3D view, where
drawing every camera's frames onto its image plane only costs.

**Timeline.** Steps are indices into the sequence's `global_time_reference`
annotation when it carries one, and plain frame indices otherwise, which
`local_frame_indices` resolves per view.

Annotations carrying a `view_id` are numbered from the start of *their own
view's* recording, so `rebase_on_global_frames` re-indexes them onto the
timeline before they are logged. Without it an AIST++ *raw* sequence puts
its 3D keypoints on steps 0-574 and its 2D keypoints on steps 775-1362,
spread differently across the nine views: measured, then fixed, then pinned
by a test. Human3.6M and EgoHumans are frame-aligned and the mapping is the
identity.

**Making the footage viewable.** Footage is shrunk by `downscale_factor`
(4 by default, 1 for the dataset's own size) and converted once, cached
beside its source as `preview_downscale_4.mp4` — named after the size it was
made for so a preview at another size cannot pick it up:

- An image sequence (EgoHumans) is encoded from its JPEGs.
- A video is transcoded, frames passed through so an index still means the
  same frame. Human3.6M needs this even at full size: it ships MPEG-4
  Part 2, and rerun decodes only H.264, H.265, AV1 and VP9. A video the
  viewer can already decode, shown at its own size, is used as it lies.

Both live in `kineo/io/ffmpeg.py` as `encode_images_to_video`,
`get_video_codec` and `transcode_video_to_h264`.

**Resizing the pixel space.** A 2D view's coordinates are the image's pixel
grid, and rerun does not stretch a child image onto the pinhole's rectangle:
`resolution` is "pixel resolution of child image space". So resized footage
alone would leave the keypoints, boxes and intrinsics marking coordinates the
frame no longer has.

`scale_pixel_space` converts all three in one pass, at the one point the
annotations enter the preview, so nothing downstream carries a scale:
`rebase_on_global_frames` works on frame indices, `keypoints_radius` reads
the resized `resolution_hw` and shrinks the dots by itself, and the logging
functions never learn a scale exists. A factor of 1 returns the mapping
untouched. The cost is that the coordinates read in the viewer are the
preview's pixels, not the dataset's.

Measured over a whole sequence at a factor of 4, against the same recording
at full size: EgoHumans 759 MB -> 34 MB, AIST++ 278 MB -> 5 MB,
Human3.6M -> 3 MB.

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

One script per dataset, each roughly 40 lines: the sequences file to read,
`--sequence` to pick one by name, `--save` to write an `.rrd` instead of
spawning the viewer, `--max-frames` to shorten the timeline; build the
dataset; call `preview_sequence`. The AIST++ variant is the listing the
caller passes, `raw` or `refined`, and Human3.6M additionally takes
`--split`.

## Testing

- `EgoHumansSequenceDataset` against a fixture directory holding a small
  `egohumans_sequences.json`, a couple of JPEGs per view, and annotation
  files: the views load, the frame timestamps follow the declared fps, and
  the annotations arrive keyed by kind.
- `local_frame_indices` and `rebase_on_global_frames`, on tensors, with no
  recording stream open: one instant lands on one step in every view, and
  annotations outside the annotated window are dropped.
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
- Previews are written into the dataset directories: 713 MB across the eight
  views of one EgoHumans sequence, four files for one Human3.6M sequence.
  They are reused on later runs and are safe to delete.
- A recording holds every view's footage in full, so previewing a many-view
  sequence costs hundreds of megabytes even with `max_frames` set.
