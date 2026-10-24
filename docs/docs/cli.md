---
title: Command line
---

# Command line

## Embed a dataset in a project

`LichtFeld-Studio licht <project.licht> --embed` embeds the dataset currently referenced by a project. To select a
different dataset, provide its path after `--embed`:

```sh
LichtFeld-Studio licht scene.licht --embed
LichtFeld-Studio licht scene.licht --embed ./dataset
```

Relative dataset paths are resolved from the current working directory. COLMAP datasets with a supported sparse or
flat layout and Transforms datasets are accepted. The project is updated in place; the source dataset is left on disk.

Run `LichtFeld-Studio licht --help` for command usage.

## Sharpen training images

`--sharpen-amount` sharpens the training images as they are loaded. [Sharpen training images](features/sharpening.md)
explains the kernels, the other `--sharpen-*` options and how sharpening affects evaluation.

## Reduce color variance

`--color-variance-weight` adds the SteadySplats color variance loss, which lowers the noise of a model in stochastic
renderers. [Color variance loss](features/color-variance.md) explains the loss, `--color-variance-warmup-iters` and the
settings it cannot be combined with.

## Train a region of interest

`--cropbox` limits training to a box, `--cropbox-rotation` turns it, and `--crop` removes everything outside it when
training completes and from exports. [Train a region of interest](features/region-of-interest.md) describes the box and
its rotation, the ROI training options and tiled training.

## Generate SuperSplat voxel collision

`voxelize` reads a Gaussian splat through the existing LFS loaders and writes a
SuperSplat `.voxel.json` / `.voxel.bin` pair. It runs without a GUI or training.

```sh
LichtFeld-Studio voxelize scene.ply -o scene.voxel.json
LichtFeld-Studio voxelize scene.sog scene.voxel.json --auto
LichtFeld-Studio voxelize scene.spz -o scene.voxel.json \
  --filter connected --seed-pos 0,1,0 --cluster-resolution 0.5
```

| Option | Default | Effect |
| --- | --- | --- |
| `--voxel-size` | `0.05` | Exact voxel edge length in input scene units. |
| `--voxel-opacity` | `0.1` | Threshold for accumulated voxel opacity; overlapping faint Gaussians can jointly occupy a voxel. |
| `--auto` | Off | Estimate cell size from Gaussian footprint and count. Cannot be combined with `--voxel-size`. |
| `--filter` | `cleanup` | `cleanup` removes exceptional area/volume outliers; `connected` also selects an observed source region; `none` retains all source Gaussians. |
| `--max-gaussian-area` | `0` | Three-sigma area cutoff in squared scene units; `0` uses a statistical cutoff. Requires cleanup or connected filtering. |
| `--seed-pos` | `0,0,0` | Region seed in input coordinates. Requires connected filtering. |
| `--cluster-resolution` | `1` | Source connectivity cell size in scene units. Requires connected filtering; does not change final voxel size. |
| `-y`, `--overwrite` | Off | Replace both output companions without a prompt. |

Distances are scene units, not inferred metres. Connected filtering uses all 26
neighbouring source-centre cells; it does not connect regions through a large
Gaussian's support box. An empty seed selects a nearby supported component.
Cleanup can reject legitimate multi-scale geometry; use `--filter none` to bypass
it. Auto is an estimate, not a memory guarantee; explicit resolutions are never
silently coarsened. The input splat stays unchanged.

The command uses CUDA voxelization, sparse occupancy, and CPU preparation and
assembly. Progress names the current operation; percentages are not time estimates.
Ctrl+C cancels at an operation boundary. Failed/cancelled writes remove temporary
files and preserve previous output companions; the metadata is published last.
Input and output paths must differ, including hard links.

Upload both files to SuperSplat. Export uses the same coordinate convention as the
SuperSplat PLY import (`Rz(180°)`), preserving input scale. Datasets, point clouds,
meshes and `.licht` projects are not voxel inputs. LOD inputs use original leaf
Gaussians; partially resident out-of-core models fail explicitly rather than using
a coarse representation. Format availability follows the existing loaders.

Limits include 32M source Gaussians, 64M compressed CUDA output records,
512,000,001 CPU tree words, and 2²¹ cells per axis. The output format also has a
24-bit node-index limit. Budget errors identify the failed operation; increase
cell size or use cleanup/connected filtering when appropriate.

Run `LichtFeld-Studio voxelize --help` for command usage. See the
[SuperSplat voxel format](https://developer.playcanvas.com/user-manual/splat-transform/voxel-format/)
for the output schema.
