# CHiME-6 dataset module

Move CHiME-6 out of `experiments/` and into `kineo/datasets/chime6/`, the way
AIST++ was ported: a download module, a preprocessing module, and a dataset
class the evaluation reads instead of owning its own corpus knowledge.

## Why it is not a `KeypointsSequenceDataset`

CHiME-6 ships no video. `ViewInput` requires a `FrameSequenceLoader`, and
relaxing that to `| None` would touch every dataset and every pipeline stage
that assumes frames exist, to serve one corpus that will never have them.
CHiME-6 therefore gets its own type. It is not a pose benchmark: it is the
ground truth for temporal calibration, because its devices are already
synchronized and their positions are known, so a de-sync can be injected and
the estimate checked against it.

## Layout

```
kineo/datasets/chime6/
  __init__.py
  chime6_device_positions.json    moved from experiments/
  chime6_download.py
  chime6_preprocess.py
  chime6_dataset.py
scripts/download_chime6_dataset.py
scripts/preprocess_chime6_dataset.py
kineo/io/audio_loader.py          + WaveformAudioLoader
```

`experiments/chime6_window_selection.py` is deleted; its logic is the
preprocessing. `experiments/chime6_audio_sync_eval.py` keeps only what is an
experiment: injected offsets, hop sweep, aggregation and CSV.

## Views

One view per Kinect array, `U01`…`U06`, reading channel `CH1`. The four
channels of one array are sample-synchronized and span 22.6 cm, so an
intra-array pair measures the estimator's noise floor rather than a
calibration case. Dropping it takes a session from 21 pairs to 15 and removes
the `pair_kind` axis entirely. The binaural `P##` microphones are not views:
they move with their wearer and have no position.

## Download

OpenSLR resource 150 publishes CHiME-6 under CC BY-SA 4.0 with no
registration, on three mirrors:

```
https://openslr.trmal.net/resources/150/
https://openslr.elda.org/resources/150/
https://openslr.magicdatatech.com/resources/150/
```

`download_chime6(output_dir, splits=("dev",), force_download=False,
skip_extract=False)` fetches `CHiME6_{split}.tar.gz` plus
`CHiME6_transcriptions.tar.gz` and `CHiME6_floorplans.tar.gz`, trying each
mirror in turn, then extracts in place. Mirror fallback stands in for the
retry loop AIST++ needs against its single host. Extraction belongs to the
download rather than the preprocessing because the tarballs are unusable
unextracted, and because it must not flatten the archive's nested
`transcriptions/transcriptions/dev` directory, which the preprocessing reads
by that path.

## Preprocessing

Relocation, not a rewrite. `parse_time`, `load_utterances`,
`concurrency_segments`, `composition`, `session_duration`, `select_cell` and
`build_windows` move over unchanged, along with the window lengths, the
class thresholds, `MAX_MUTUAL_OVERLAP` and the 5 s `AUDIO_START_GUARD_S` that
keeps windows clear of the opening sync beep.

Two changes. The `class` key becomes `content_class`, since the dataset
exposes it as a field and `class` is a keyword. And every window gains its
views, resolved from `chime6_device_positions.json`: the units flagged
`has_audio`, each with the path of its `CH1` wav relative to the dataset root,
its floorplan position in metres and its room. Baking the geometry into the
window makes the positions file a preprocessing input rather than a runtime
dependency of whatever reads the dataset.

Output is `<dataset_dir>/chime6_windows.json`:

```json
{
  "meta": {"sessions": ["S02"], "lengths_s": [30.0], "report": {}},
  "windows": [
    {
      "window_id": "S02_L30_single_00",
      "session_id": "S02",
      "start_time_s": 1234.5,
      "duration_s": 30.0,
      "content_class": "single",
      "composition": {"silence": 0.2, "single": 0.75, "overlap": 0.05},
      "n_distinct_speakers": 2,
      "max_concurrent_speakers": 1,
      "speakers": ["P05", "P06"],
      "selection": {"target_class": "single", "relaxed": false,
                    "rank": 0, "purity": 0.75},
      "views": {
        "U01": {"audio_path": "CHiME6/audio/dev/S02_U01.CH1.wav",
                "position_m": [2.987, 1.851], "room": "kitchen"}
      }
    }
  ]
}
```

`meta.report` records, per cell, how many candidates qualified and why a cell
had to fall back to `mixed`.

## Dataset

```python
class AudioView(TypedDict):
    view_id: str
    audio_path: str          # absolute
    audio_loader: AudioLoader
    position_m: tuple[float, float]
    room: str

class AudioWindow(TypedDict):
    window_id: str
    session_id: str
    start_time_s: float
    duration_s: float
    content_class: str
    composition: dict[str, float]
    n_distinct_speakers: int
    max_concurrent_speakers: int
    speakers: list[str]
    selection: dict
    views: list[AudioView]

class CHiME6AudioDataset:  # __len__, __getitem__, __iter__
```

`audio_loader` is a `WaveformAudioLoader` already scoped to the window, which
is the ordinary read. `audio_path` stays on the view because the evaluation
does something the ordinary read cannot express: it reads the target device at
a deliberately shifted position, and builds its own loader to do so.

`WaveformAudioLoader(audio_path, start_frame=0, n_frames=-1, device)` is a
thin `AudioLoader` over `kineo.io.audio_file.load_waveform`.

## Evaluation

`chime6_audio_sync_eval.py` opens the dataset and, per window, pairs its views
with `itertools.combinations`, taking the distance from `math.hypot` on
`position_m` and room agreement from `room`. `build_pairs`, `load_window`,
`INTRA_ARRAY_DISTANCE_M`, the `pair_kind` column, the `by_pair_kind` summary
and the hardcoded 16 kHz sample rate all disappear. The bounds and beep-guard
check stays here rather than moving into the loader: the preprocessing already
guarantees every window is in bounds past the guard, so only an injected
offset can read off the end.

## Tasks

`chime6-windows` is replaced by `chime6-download` and `chime6-preprocess`.
`chime6-sync-eval` keeps its arguments, with `windows_json` defaulting to
`<dataset_dir>/chime6_windows.json`.

## Testing

`tests/datasets/test_chime6_preprocess.py` covers the pure functions on
synthetic arrays, with no corpus on disk: `parse_time`;
`concurrency_segments` on overlapping, adjacent and disjoint utterances;
`composition` for rows summing to one and a hand-computed split; and
`select_cell` for rejecting a candidate that overlaps a selection past
`MAX_MUTUAL_OVERLAP` and for ranking a qualifying candidate above a
higher-scoring one that does not qualify.

`tests/pipeline/test_mfcc_temporal_calibration.py` is untouched.
