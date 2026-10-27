# MVS mask geometry core

This package maps binary labels from a photo-io canonical camera into the existing undistorted camera produced by the accepted MVS bridge. It does not generate masks, rerun SfM, select instances, or run MVS.

`warp_binary_mask(mask, source_camera, target_camera)` takes uint8 H×W labels 0/255 and existing pycolmap camera objects. The caller must first bind frame/image identity and verify the source and target poses refer to the same image and camera axes. Target must be PINHOLE/SIMPLE_PINHOLE; dimensions and complete source distortion parameters are used, with no camera mutation.

The returned `WarpedMask` contains a binary mask, a separate 0/255 validity mask, and counts. Destination centers are column+0.5,row+0.5; inverse target rays are projected through the source camera, and labels are sampled discretely. Nonfinite/outside mappings are zero in both outputs. Valid zero pixels are known background, invalid zero pixels are unknown; never conflate them. Validity describes nearest label raster support, not an RGB bilinear footprint or geometric accuracy.

Geometry follows the pixel coordinate convention of [COLMAP 4.2.1 warp.cc](https://raw.githubusercontent.com/colmap/colmap/4.2.1/src/colmap/image/warp.cc). No extra OpenMVS half-pixel adjustment belongs here. This is an independent NumPy implementation using pycolmap's projection APIs, with no copied upstream source.

Validation: 15 tiny native-camera cases passed once; a separate synthetic checkerboard probe used three existing 44-view camera pairs (indices3/43/65) and preserved all input model bytes. Source cameras were1024×683 SIMPLE_RADIAL; target cameras1013×675 PINHOLE. These results validate neither segmentation nor OpenMVS consumption. The local evidence bundle is intentionally excluded from Git.

File/manifest handling and OpenMVS integration are delegated separately under docs/contracts_mvs_masks_v1.md. Keep this tested core read-only in that task; report a counterexample if a core change is necessary.
