import os, sys, glob, numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

TARGET = (96, 128, 96)

def crop_and_save(fname: str):
    try:
        img  = nib.load(fname)
        data = img.get_fdata(dtype=np.float32)

        if data.ndim not in (3, 4):
            raise ValueError(f"ndim {data.ndim} unsupported")

        dx, dy, dz = data.shape[:3]
        tx, ty, tz = TARGET
        if dx < tx or dy < ty or dz < tz:
            raise ValueError(f"too small: {(dx,dy,dz)} < {(tx,ty,tz)}")

        # center crop indices
        sx = (dx - tx) // 2; ex = sx + tx
        sy = (dy - ty) // 2; ey = sy + ty
        sz = (dz - tz) // 2; ez = sz + tz

        slc = (slice(sx, ex), slice(sy, ey), slice(sz, ez))
        if data.ndim == 4:
            slc = (*slc, slice(None))

        cropped = data[slc]

        # adjust affine: we removed voxels at the "start" of each axis
        affine = img.affine.copy()
        affine[:3, 3] += affine[:3, :3] @ np.array([sx, sy, sz], dtype=float)

        nib.save(nib.Nifti1Image(cropped, affine, header=img.header), fname)
        return (fname, None)
    except Exception as e:
        return (fname, str(e))

def main(src_dir, n_jobs=None):
    # make BLAS libraries single-threaded to avoid fork/thread crashes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    files = sorted(glob.glob(os.path.join(src_dir, "**", "*.nii*"), recursive=True))
    if not files:
        sys.exit("No NIfTI files found.")

    errors = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(crop_and_save, f): f for f in files}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Center-cropping to 96×128×96"):
            fname, err = fut.result()
            if err:
                errors.append((fname, err))

    if errors:
        print("\n---- Errors ----")
        for f, e in errors[:20]:
            print(f"* {f}: {e}")
        print(f"... and {max(0, len(errors)-20)} more.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.exit("Usage: python crop_nifti_safe.py /data/dir [n_jobs]")
    src_dir = sys.argv[1]
    jobs = int(sys.argv[2]) if len(sys.argv) == 3 else None
    main(src_dir, jobs)
