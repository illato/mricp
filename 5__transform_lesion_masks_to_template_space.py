#!/usr/bin/env python
# coding: utf-8

# - Running via `nbconvert --to script` per multiprocessing
#   - 8 tasks/processes
#     - (`#SBATCH --ntasks=8`)
#   - 4 threads per task
#     - (`#SBATCH --cpus-per-task=4`)
#   - 64GB RAM (to be more than safe)
#     - (`#SBATCH --mem=64GB`)
#   
# (if reproducing, adjust as necessary)

# ---

# `conda activate mri`
# 
# ---
# 
# `jupyter nbconvert --to script <__this notebook's name__>.ipynb --execute`

# ---

# ---

# # Transform Lesion Masks to Template Space

# ---

# In[1]:


import glob
import os
import multiprocessing
from subprocess import run

def process_scan(args):
    lesion_mask, affine_transform = args

    if not os.path.exists(affine_transform):
        print(f"[SKIP] Missing transform: {affine_transform}")
        return
    
    # Set environment variables for multi-threading
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'  # Limits OpenMP threads
    env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '4'  # Limits ITK threads

    lesion_mask = os.path.expanduser(lesion_mask)
    affine_transform = os.path.expanduser(affine_transform)
    output_dir = os.path.dirname(lesion_mask)
    
    output_mask_transformed = os.path.join(
        output_dir,
        os.path.basename(lesion_mask).replace('.nii.gz', '').replace('.nii', '') + '_rigid_MIWarped.nii.gz'
    )
    
    # Reference image used in registration
    reference_image = os.path.expanduser(
        '~/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz'
    )
    
    # Construct the antsApplyTransforms command
    apply_command = [
        "antsApplyTransforms",
        "-d", "3",                                     # 3D image
        "-i", lesion_mask,                             # Input image (mask)
        "-r", reference_image,                         # Reference image (target space)
        "-t", affine_transform,                        # Rigid/affine transform
        "-o", output_mask_transformed,                 # Output filename
        "-n", "NearestNeighbor"                        # Important: use nearest neighbor for masks
    ]
    
    # Run the command
    print(f"Transforming mask to template space: {os.path.basename(lesion_mask)}...")
    result = run(apply_command, capture_output=True, text=True, env=env)
    
    # Check for errors
    if result.returncode != 0:
        print(f"[ERROR] {lesion_mask}:\n{result.stderr}")
    else:
        print(f"[OK] {lesion_mask} → {output_mask_transformed}")

def main():
    # ISBI dataset (2:1 masks : .mat)
    files_isbi = glob.glob(os.path.expanduser(
        '~/dissertation/data/MRI/ISBI/training/training*/masks/*.nii'
    ), recursive=True)

    isbi_pairs = []
    for mask_path in files_isbi:
        mat_path = (
            mask_path
            .replace('masks', 'orig')
            .replace('mask1.nii', 't2_rigid_MI0GenericAffine.mat')
            .replace('mask2.nii', 't2_rigid_MI0GenericAffine.mat')
        )
        isbi_pairs.append((mask_path, mat_path))

    # Muslim dataset (1:1 masks : .mat)
    files_muslim = glob.glob(os.path.expanduser(
        '~/dissertation/data/MRI/Muslim_et_al/Patient-*/*[0-9]-LesionSeg-T2_oriented_z.nii'
    ), recursive=True)

    muslim_pairs = []
    for mask_path in files_muslim:
        mat_path = (
            mask_path
            .replace('LesionSeg-', '')
            .replace('.nii', '_rigid_MI0GenericAffine.mat')
        )
        muslim_pairs.append((mask_path, mat_path))

    # Combine both datasets
    all_pairs = isbi_pairs + muslim_pairs

    # Use multiprocessing to apply transforms
    num_processes = 8
    multiprocessing.set_start_method('fork', force=True)

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_scan, all_pairs)

if __name__ == '__main__': # best practice to prevent execution on import
    print('STARTING')
    main()
    print('ENDING')


# ---

# ---

# ---
