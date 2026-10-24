# Voxelization timing

Build the optional generation benchmark:

```sh
cmake --build build --target bench_collision_generation
build/tests/bench_collision_generation snapshot.f32 0.05 auto
```

A frozen snapshot consists of native `CollisionGaussian` rows: center XYZ, three
column-major one-sigma axis vectors, and opacity (13 float32 values per row).
Modes are `none`, `cleanup`, `connected`, and `auto`; optional seed XYZ follows.
The benchmark excludes source parsing, source extraction, and output writing.
`LFS_COLLISION_BENCHMARK_TREE=/tmp/tree.u32` saves the canonical tree outside the
reported interval for exact before/after comparison; this is a native test dump,
not a project or interchange format.

For complete I/O timing, measure the actual `LichtFeld-Studio voxelize` command.
Use identical sources/settings and alternating trials with other GPU work idle.
Report generation time separately from total wall time and peak RSS. Output
counts alone do not establish occupancy equality.
