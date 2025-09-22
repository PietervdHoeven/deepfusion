import sys
import nibabel as nib

def main(nifti_path):
    img = nib.load(nifti_path)
    print("Shape:", img.shape)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_nifti_shape.py <nifti_file>")
        sys.exit(1)
    main(sys.argv[1])