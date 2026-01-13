
import os
import glob
import matlab.engine

# Start MATLAB engine once.
print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()

# Add the directory containing MATLAB functions to the MATLAB path.
functions_dir = os.path.expanduser('~/dissertation/data/MRI/matlab_functions')
eng.addpath(functions_dir, nargout=0)

# Expand the root directory.
orig_root = os.path.expanduser('~/dissertation/data/MRI/')

# Get list of all matching directories.
matching_dirs = [
    d for d in glob.glob(os.path.join(orig_root, '_[MH]*/*'), recursive=True)
    if os.path.isdir(d)
]

# Filter folders by matching substrings.
slice_folders_ms_isbi_ph3_test =    [x for x in matching_dirs if '_MS__ISBI_3T_test' in x]
slice_folders_ms_isbi_ph3_train =   [x for x in matching_dirs if '_MS__ISBI_3T_train' in x]
slice_folders_ms_muslim_15t       = [x for x in matching_dirs if '_MS__Muslim_et_al_15T' in x]
slice_folders_healthy_ph3         = [x for x in matching_dirs if '_Healthy__IXI_3T' in x]
slice_folders_healthy_ph15        = [x for x in matching_dirs if '_Healthy__IXI_15T_Guys' in x]
slice_folders_healthy_ge15        = [x for x in matching_dirs if '_Healthy__IXI_15T_IOP' in x]

# Combine all folder lists.
all_folders = (
    slice_folders_ms_isbi_ph3_test +
    slice_folders_ms_isbi_ph3_train +
    slice_folders_ms_muslim_15t +
    slice_folders_healthy_ph3 +
    slice_folders_healthy_ph15 +
    slice_folders_healthy_ge15
)

# Define sigma values for blur.
sigma_values = [1, 2, 3, 4]

# Define the contrast operations as tuples:
# (readable name, MATLAB function name, destination folder suffix)
contrast_operations = [
    ('imadjust', 'applyImadjust', '_contrast_imadjust'),
    ('histeq', 'applyHisteq', '_contrast_histeq'),
    ('adapthisteq', 'applyAdapthisteq', '_contrast_adapthisteq')
]

# Loop over each folder and process all PNG files.
for folder in all_folders:
    png_files = glob.glob(os.path.join(folder, '*.png'))
    for png_file in png_files:
        # --- Blur Operations ---
        for sigma in sigma_values:
            # Compute the relative path.
            relative_path = os.path.relpath(folder, orig_root)
            # Destination root includes the sigma value.
            dest_root = os.path.join(orig_root, f'_blurred_SD{sigma}')
            # Destination folder mirrors the original folder structure.
            dest_folder = os.path.join(dest_root, relative_path)
            os.makedirs(dest_folder, exist_ok=True)
            
            # The output filename is the same as the input filename.
            filename = os.path.basename(png_file)
            dest_file = os.path.join(dest_folder, filename)
            
            print(f"Blurring {png_file.split('/data/')[1]} with sigma={sigma}")
            print(f"Saving to {dest_file.split('/data/')[1]}")
            
            # Call the MATLAB blur function.
            eng.applyGaussian(png_file, dest_file, sigma, nargout=0)
        
        # --- Contrast Operations ---
        for op_name, matlab_func, folder_suffix in contrast_operations:
            # Compute the relative path.
            relative_path = os.path.relpath(folder, orig_root)
            dest_root = os.path.join(orig_root, folder_suffix)
            dest_folder = os.path.join(dest_root, relative_path)
            os.makedirs(dest_folder, exist_ok=True)
            
            filename = os.path.basename(png_file)
            dest_file = os.path.join(dest_folder, filename)
            
            print(f"Applying {op_name} on {png_file.split('/data/')[1]}")
            print(f"Saving to {dest_file.split('/data/')[1]}")
            
            # Dynamically get the function from the MATLAB engine and call it.
            func = getattr(eng, matlab_func)
            func(png_file, dest_file, nargout=0)
            

# When finished, quit MATLAB.
print("Processing complete. Quitting MATLAB engine.")
eng.quit()
