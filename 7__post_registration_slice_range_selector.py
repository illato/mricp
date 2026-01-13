#conda activate mri

import glob
import os
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def interactive_mri_visualization(scan, slice_positions, selected_scans):
    """
    Displays one pair of low/high MRI slices (from the given scan) at a time.
    
    - scan: a file path to the scan (or a nibabel image object; here we assume a path)
    - slice_positions: list of fractional positions (or absolute indices) for the low slice.
      (For fractional positions, the low slice is computed as int(total_slices * pos).)
    - selected_scans: a dict that maps the slice position index (0...len(slice_positions)-1)
      to a list of scan paths that were “accepted” when that slice was shown.
    """
    # Load the MRI scan using nibabel.
    img = nib.load(scan)
    data = img.get_fdata()
    num_slices = data.shape[2]
    
    # Compute the low and high slice indices for each slice position.
    low_indices = []
    high_indices = []
    for pos in slice_positions:
        # If pos < 1 assume it is a fraction, otherwise assume it is an absolute index.
        if pos < 1:
            low_idx = int(num_slices * pos)
        else:
            low_idx = int(pos)
        # High slice is 42 slices above the low slice (or the last slice, if that goes out of bounds)
        high_idx = min(low_idx + 42, num_slices - 1)
        low_indices.append(low_idx)
        high_indices.append(high_idx)
    
    # We'll use a mutable container (a one‐element list) for the current index.
    current_index = [0]  # start with the first slice position

    # Set up the figure with two side‐by‐side subplots for the low and high slices.
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.3)  # make room for buttons

    # Display the initial slices
    im_low = axs[0].imshow(data[:, :, low_indices[current_index[0]]].T,
                           cmap='gray', origin='lower')
    axs[0].set_title(f"Low Slice: {low_indices[current_index[0]]}")
    axs[0].axis('off')

    im_high = axs[1].imshow(data[:, :, high_indices[current_index[0]]].T,
                            cmap='gray', origin='lower')
    axs[1].set_title(f"High Slice: {high_indices[current_index[0]]}")
    axs[1].axis('off')

    # Create button axes (you can adjust these positions as needed)
    ax_up = plt.axes([0.3, 0.15, 0.1, 0.075])
    ax_down = plt.axes([0.45, 0.15, 0.1, 0.075])
    ax_continue = plt.axes([0.6, 0.15, 0.1, 0.075])

    button_up = Button(ax_up, 'Up')
    button_down = Button(ax_down, 'Down')
    button_continue = Button(ax_continue, 'Continue')

    def update_display():
        """Update both subplots with the new low and high slice images."""
        idx = current_index[0]
        im_low.set_data(data[:, :, low_indices[idx]].T)
        axs[0].set_title(f"Low Slice: {low_indices[idx]}")
        im_high.set_data(data[:, :, high_indices[idx]].T)
        axs[1].set_title(f"High Slice: {high_indices[idx]}")
        fig.canvas.draw_idle()

    def on_up(event):
        """Move to the next slice position (if possible)."""
        if current_index[0] < len(slice_positions) - 1:
            current_index[0] += 1
            update_display()

    def on_down(event):
        """Move to the previous slice position (if possible)."""
        if current_index[0] > 0:
            current_index[0] -= 1
            update_display()

    def on_continue(event):
        """
        When the user clicks Continue, record the current scan path into the
        selected_scans dictionary for the current slice position and close the figure.
        """
        idx = current_index[0]
        selected_scans[idx].append(scan)
        plt.close(fig)  # close the figure to move on to the next scan

    # Connect the buttons to their callbacks
    button_up.on_clicked(on_up)
    button_down.on_clicked(on_down)
    button_continue.on_clicked(on_continue)

    plt.show()

#
#
#
if __name__ == "__main__":
    print('NOTE: Script is only processing 1/6 data subsets (modify appropriately for desired target')

    # files_healthy_ph3 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/IXI/*HH*T2_rigid_MIWarped.nii.gz'), recursive=True)
    # files_healthy_ph15 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/IXI/*Guy*T2_rigid_MIWarped.nii.gz'), recursive=True)
    # files_healthy_ge15 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/IXI/*IOP*T2_rigid_MIWarped.nii.gz'), recursive=True)    
    # shape == (256, 256, x)
    # x256_y256 = [1, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 24, 25, 27, 29, 30, 4, 44, 46, 47, 49, 5, 50, 51, 52, 58, 6, 60, 7, 8]
    # files_ms_muslim_15t = [os.path.expanduser(f'~/dissertation/data/MRI/Muslim_et_al/Patient-{i}/{i}-T2_oriented_z_rigid_MIWarped.nii.gz') for i in x256_y256]
    # files_ms_isbi_ph3 = glob.glob(os.path.expanduser('~/dissertation/data/MRI/ISBI/training/training*/orig/*t2_rigid_MIWarped.nii.gz'), recursive=True)
    files_ms_isbi_ph3_test = glob.glob(os.path.expanduser('~/dissertation/data/MRI/ISBI/testdata_website/test*/orig/*t2_rigid_MIWarped.nii.gz'), recursive=True)
    
    # Define slice positions. (Assuming fractional positions.)
    slice_positions = [0.3, 0.34, 0.37, 0.39, 0.4, 0.42, 0.45, 0.48, 0.5, 0.52, 0.54]

    # Prepare a dictionary to collect the selected scan paths for each slice position.
    # Each key is an index (0 through 8) corresponding to one of the slice_positions.
    selected_scans = {i: [] for i in range(len(slice_positions))}

    # Loop over each scan. For each scan, the interactive window will show the pair (low/high)
    # for one slice position at a time. The user may change the displayed slice with Up/Down,
    # and then click Continue to record the scan (for the currently shown slice) and move on.
    for scan in files_ms_isbi_ph3_test:
        interactive_mri_visualization(scan, slice_positions, selected_scans)

    # When all scans have been processed, you can inspect the dictionary.
    print("Selected scans by slice position:")
    for idx, scan_list in selected_scans.items():
        print(f"Slice position {slice_positions[idx]} (index {idx}):")
        for s in scan_list:
            print(f"  {s}")