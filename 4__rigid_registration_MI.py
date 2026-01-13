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

# ---

# # Setup

# ---

# In[ ]:


import glob
import os


# ### Muslim et al.

# In[ ]:


files_ms_muslim_15t = glob.glob(os.path.expanduser('~/dissertation/data/MRI/Muslim_et_al/Patient-*/*[0-9]-T2_oriented_z.nii'), recursive=True)


# ### ISBI

# In[ ]:


files_ms_isbi_ph3 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/ISBI/training/training*/orig/*t2.nii.gz'), recursive=True)
files_ms_isbi_ph3_test = glob.glob(os.path.expanduser('~/dissertation/data/MRI/ISBI/testdata_website/test*/orig/*t2.nii.gz'), recursive=True)


# ### IXI

# In[ ]:


files_healthy_ph3 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/IXI/*HH*T2.nii.gz'), recursive=True)
files_healthy_ph15 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/IXI/*Guy*T2.nii.gz'), recursive=True)
files_healthy_ge15 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/IXI/*IOP*T2.nii.gz'), recursive=True)


# In[ ]:


import multiprocessing
import sys

def process_scan(scan_path):
    import os
    from subprocess import run

    # Set environment variables for multi-threading
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '4'  # Limits OpenMP threads
    env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '4'  # Limits ITK threads

    scan_path = os.path.expanduser(scan_path)

    # Get scan filename without extension
    scan_name = os.path.basename(scan_path).replace(".nii.gz", "").replace(".nii", "")
    print(f"Processing scan: {scan_name}")

    # Get scan directory
    output_dir = os.path.dirname(scan_path)
    
    # Define output paths for bias-corrected image and registration
    corrected_image = os.path.join(output_dir, f"{scan_name}_N4Corrected.nii.gz")
    output_prefix = os.path.join(output_dir, f"{scan_name}_rigid_MI")
    output_warped = f"{output_prefix}Warped.nii.gz"
    output_transform = f"{output_prefix}0GenericAffine.mat"
    
    # Reference image for registration
    reference_image = os.path.expanduser(
        '~/.cache/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T2w.nii.gz'
    )
    print(output_prefix)    
    
    # Step 1: N4 Bias Field Correction
    ## https://github.com/ANTsX/ANTs/wiki/N4BiasFieldCorrection
    n4_command = [
        "N4BiasFieldCorrection",
        "-d", "3",  # 3D imagery
        "-i", scan_path,  # Input image
        "-o", corrected_image,  # Output corrected image
        "-s", "2",  # Shrink factor 
        "-c", "[50x50x50x50,0.0001]",  # Convergence criteria
        "-b", "[200]",  # spline distance for the B-spline field (defines the coarseness of the bias field estimate)
        "-v", "1",  # Verbose output
    ]

    # Run N4 Bias Correction
    print(f"Running N4BiasFieldCorrection for {scan_path}...")
    n4_result = run(n4_command, capture_output=True, text=True, env=env)
    if n4_result.returncode != 0:
        print(f"Error in N4BiasFieldCorrection for {scan_path}:")
        print(n4_result.stderr)
        return  # Skip registration if bias correction fails
    else:
        print(f"Bias-corrected image saved to {corrected_image}.")

    
    # Step 2: Rigid Registration
    ants_command = [
        "antsRegistration",                                                        # ANTs registration command
        "--dimensionality", "3",                                                   # 3D imagery
        "--float", "0",                                                            # Use double precision
        "--output", f"[{output_prefix},{output_warped}]",                          # Output prefix
        "--interpolation", "Linear",                                               # Interpolation method
        "--winsorize-image-intensities", "[0.005,0.995]",                          # Intensity normalization
        "--use-histogram-matching", "1", # Histogram matching helps align distribution differences caused by non-brain tissue, making similarity metric more effective
        "--initial-moving-transform", f"[{reference_image},{corrected_image},1]",  # Initial transform
        "--transform", "Rigid[0.1]",                                               # Rigid transformation
        # Cross Correlation metric should work better than Mutual Information 
        #       when same (e.g., T2-w) modality (according to documentation); 
        #       however, on average across the different 1.5T/3T scans, MI 
        #       appears to result in more consistent head positioning 
        "--metric", f"MI[{reference_image},{corrected_image},1,32,Regular,0.25]",  # Mutual information metric
        "--convergence", "[2000x1000x500x250,1e-6,10]",  # Convergence criteria (increased iterations to help account for non-brain tissue) 
        "--shrink-factors", "12x8x4x2",      # Multi-resolution levels (increased to handle larger, more complex images due to inclusion of skull/dura)
        "--smoothing-sigmas", "4x3x2x1vox",  # Smoothing at each level (increased to handle larger, more complex images due to inclusion of skull/dura)
        "--verbose", "1"
    ]
    
    # Run the command
    print(f"Registering {corrected_image}...")
    result = run(ants_command, capture_output=True, text=True, env=env)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error with {corrected_image}:")
        print(result.stderr)
    else:
        print(f"Successfully registered {corrected_image}. Outputs saved to {output_dir}.")

def main():
    # List of T2 scan paths
    t2_scans = [
        *files_ms_isbi_ph3_test,
        *files_ms_isbi_ph3,
        *files_ms_muslim_15t,
        *files_healthy_ph3,
        *files_healthy_ph15,
        *files_healthy_ge15
    ]

    # Use multiprocessing Pool to process scans in parallel
    num_processes = 8  # Number of parallel processes

    # Handle Jupyter notebook environment for multiprocessing
    multiprocessing.set_start_method('fork', force=True) # similar to 'spawn' on Windows

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_scan, t2_scans)

if __name__ == '__main__': # best practice to prevent execution on import
    main()


# ---

# ---

# ---
