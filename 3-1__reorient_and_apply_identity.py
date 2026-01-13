
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Button
import glob
import os

# Function to display the middle slice with an interactive correction option
def display_and_correct(file_list):
    for file_path in file_list:
        img = nib.load(file_path)
        global data
        data = img.get_fdata()

        mask_file = file_path.replace("-T2", "-LesionSeg-T2")
        assert os.path.exists(mask_file)
        mask_img = nib.load(mask_file)
        global mask_data
        mask_data = mask_img.get_fdata()

        # Display the middle slice
        mid_slice = data.shape[2] // 2
        fig, ax = plt.subplots()
        ax.imshow(data[:, :, mid_slice], cmap="gray")
        ax.set_title(f"File: {file_path.split('/')[-1]} - Middle Slice")
        plt.axis("off")

        # replace invalid 0.0 value with identity
        img.header['quatern_d']=1.0
        mask_img.header['quatern_d']=1.0
        # replace 'unknown' with 'scanner'
        img.header['qform_code']=1
        mask_img.header['qform_code']=1
        
        # Check if file already ends with "_oriented" to avoid duplicating it
        if file_path.endswith(".nii.gz") and not file_path.endswith("_oriented.nii.gz"):
            corrected_path = file_path.replace(".nii.gz", "_oriented.nii.gz")
        elif file_path.endswith(".nii") and not file_path.endswith("_oriented.nii"):
            corrected_path = file_path.replace(".nii", "_oriented.nii")
        else:
            corrected_path = file_path

        # Function to apply the rotation and save the corrected imagery/mask data
        def apply_rotation(event):
            global data, mask_data
            data = np.rot90(data, k=-1, axes=(0, 1))  # Rotate main image 90° clockwise
            mask_data = np.rot90(mask_data, k=-1, axes=(0, 1))
            continue_to_next(None)

        # Function to move to the next file
        def continue_to_next(event):
            corrected_img = nib.Nifti1Image(data, img.affine, img.header)
            nib.save(corrected_img, corrected_path)
            print(f"Saved corrected file as {corrected_path}")
        
            # Process and save the corresponding mask file
            if mask_file.endswith(".nii.gz") and not mask_file.endswith("_oriented.nii.gz"):
                corrected_mask_path = mask_file.replace(".nii.gz", "_oriented.nii.gz")
            elif mask_file.endswith(".nii") and not mask_file.endswith("_oriented.nii"):
                corrected_mask_path = mask_file.replace(".nii", "_oriented.nii")
            else:
                corrected_mask_path = mask_file
            corrected_mask_img = nib.Nifti1Image(mask_data, mask_img.affine, mask_img.header)
            nib.save(corrected_mask_img, corrected_mask_path)
            print(f"Saved corrected mask file as {corrected_mask_path}")
            plt.close(fig)  # Close current figure to move to the next file

        # Add "Apply" button and link to apply_rotation function
        ax_apply = plt.axes([0.7, 0.02, 0.1, 0.05])
        btn_apply = Button(ax_apply, 'Apply')
        btn_apply.on_clicked(apply_rotation)

        # Add "Continue" button and link to continue_to_next function
        ax_continue = plt.axes([0.81, 0.02, 0.1, 0.05])
        btn_continue = Button(ax_continue, 'Continue')
        btn_continue.on_clicked(continue_to_next)

        plt.show()

files_ms_muslim_15t = glob.glob(os.path.expanduser('~/dissertation/data/MRI/Muslim_et_al/Patient-*/*[0-9]-T2.nii'), recursive=True)
display_and_correct(files_ms_muslim_15t)