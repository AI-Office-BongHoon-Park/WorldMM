# Kitchen Gaussian Splatting PoC Status

Date: 2026-05-19 KST
Owner: WorldMM spatial-rendering PoC
Place: `A1_JAKE/DAY1/kitchen`

## Outcome

This PoC did not produce a real trained Gaussian Splatting model.
The shipped artefact is an honest depth-warp fake fallback.
The render exists at `output/gs_renders/A1_JAKE/DAY1/kitchen_view_01.png`.
The PNG metadata labels it `depth-warp fake; NOT real Gaussian Splatting`.

## Files Written

`tools/gs_render_kitchen.py` implements the `gs_render(place, pose)`-shaped CLI.
`output/gs_models/A1_JAKE/DAY1/kitchen.ply` is a proxy RGB-D point cloud, not GS.
`output/gs_models/A1_JAKE/DAY1/kitchen.ply.status.json` records failure details.
`output/gs_renders/A1_JAKE/DAY1/kitchen_view_01.png` is the 480px-wide fallback render.

## Input Media

Kitchen metadata was found in `spatial_extraction_results_chatgpt-gpt-5.4.json`.
Useful chunks include `112093000`, `113293000`, `113342900`, and `119113000`.
The fallback uses `output/thumbnails/A1_JAKE/DAY1/thumb_113355400.jpg`.
Candidate kitchen clip `DAY1_A1_JAKE_11353000.mp4` exists and is 600 frames at 20 fps.
Only one frame/thumbnail was used for the final fallback, below the 30-frame cap.

## Attempt 1: Full gsplat Training

Initial `nvidia-smi`: total 6144 MiB, used 965 MiB, free 4807 MiB.
Torch CUDA smoke: free 4,949,475,328 bytes, total 6,050,938,880 bytes.
Installed `gsplat==1.5.3` using `uv pip install`.
`import gsplat` succeeded and reported version `1.5.3`.
Training still blocked before optimization.
`colmap -h` failed with `colmap: command not found`.
No calibrated camera poses/intrinsics were available for 3DGS training.
Under §2.2, the pipeline requires COLMAP/VGGT before `gsplat` training.
No 30-minute training run was started because prerequisites were absent.

## Attempt 2: instant-ngp / Smaller Variant

`instant-ngp --help` failed with `instant-ngp: command not found`.
`uv pip install 'instant-ngp==0.1.0'` failed.
The resolver reported no registry package named `instant-ngp`.
Installing from source would require CUDA build tooling and was outside tiny PoC scope.
This route was therefore blocked before training.

## Attempt 3: Depth Anything V2 Fallback

`transformers` depth pipeline loaded `depth-anything/Depth-Anything-V2-Small-hf`.
Inference ran on `cuda:0` against the kitchen thumbnail.
The model returned a relative depth tensor of shape `479 x 479`.
The CLI resized the source to 480px wide, normalized depth, then applied a 24px lateral warp.
A small blend with the source frame fills warp tears.
The result is useful for PPT illustration only, not valid novel-view GS evidence.

## VRAM Measurements

Before install/probes: 6144 MiB total, 965 MiB used, 4807 MiB free.
Torch CUDA probe: 4.95 GB free, 6.05 GB total bytes visible to PyTorch.
After `gsplat`, `instant-ngp`, and depth smoke tests: 6144 MiB total, 965 MiB used, 4807 MiB free.
Depth Anything V2 Small fit inside the available budget for one 480px-ish image.
Real GS memory was not measured because pose generation and trainer prerequisites failed first.

## Path Forward

Install COLMAP or run VGGT to generate camera intrinsics/extrinsics for <=30 kitchen frames.
Keep the frame cap at 1 fps and one 30-second kitchen clip for the first real GS trial.
Use a minimal 3DGS trainer that accepts COLMAP output and exports `.ply`.
Cap optimization at 30 minutes wall-clock on the RTX 3050.
If VRAM OOM occurs during real training, reduce image size before reducing frame count.
Replace `kitchen.ply` with a real Gaussian splat only after render verification.
Until then, any slide using `kitchen_view_01.png` must label it depth-warp fake.
