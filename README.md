# 📊 Reliable Uncertainty Under Class Imbalance and Distribution Shift: Class-Conditional Conformal Prediction for Multiple Sclerosis Diagnosis  
**Alexander S. Millar MS; Ramkiran Gouripeddi MBBS, MS, FAMIA; Julio C. Facelli PhD, FACMI**  
Department of Biomedical Informatics, University of Utah  
Correspondence: u0740821@umail.utah.edu  

---

## Overview  
The notebooks in this repository contain the full data‐analysis pipeline supporting "**Reliable Uncertainty Under Class Imbalance and Distribution Shift: Class-Conditional Conformal Prediction for Multiple Sclerosis Diagnosis**" (JHIR, 2026).  
All code is reproducible and was executed in the Python environments specified in `0_setup.ipynb`.

---

## Repository Structure

### Summary Statistics
- **Total Python files:** 7
- **Total Jupyter notebooks:** ~20
- **Local modules:** 2 (`conformal.py`, `util.py`)
- **MATLAB functions:** 4
- **Model files:** 1 (`.keras`)

### Core Modules
- **`conformal.py`** - Conformal prediction implementation (depends on `numpy`)
- **`util.py`** - Utility functions for data loading, preprocessing, and prediction (depends on `numpy`, `pandas`, `tensorflow`, `PIL`)

### MATLAB Functions (`matlab_functions/`)
- `applyGaussian.m` - Gaussian blur operations
- `applyImadjust.m` - Contrast adjustment
- `applyHisteq.m` - Histogram equalization
- `applyAdapthisteq.m` - Adaptive histogram equalization

---

## Setup & Installation

### Prerequisites
1. **Python Environment** - See `0__setup.ipynb` for detailed environment setup
   - Primary environment: `mri` (conda)
   - Classification environment: `ms_classification` (conda)
   - MATLAB environment: `matlab_env` (venv)

2. **MATLAB R2023b** - Required for data augmentation (`8__matlab_execute.py`)
   - Install MATLAB Engine for Python: `pip install matlabengine==23.2.3`

3. **ANTs (Advanced Normalization Tools)** - Required for image registration
   - Used in: `4__rigid_registration_MI.py`, `5__transform_lesion_masks_to_template_space.py`

4. **TemplateFlow** - Download MRI template for registration
   - Installed via: `pip install templateflow`
   - Template location: `~/.cache/templateflow/tpl-MNI152NLin2009cAsym/`

### External Data Requirements
The following data must be obtained separately (not included in repository):
- **MRI NIfTI files** - T2-weighted scans from:
  - Muslim et al. dataset: `./Muslim_et_al/` [(link)](https://doi.org/10.17632/8bctsm8jz7.1)
  - ISBI dataset: `./ISBI/` [(link)](https://smart-stats-tools.org/lesion-challenge)
  - IXI dataset: `./IXI/` [(link)](http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar)  

The following data is produced as output from the above:
- **MRI slice PNGs** - Processed slice images in variant folders: `./_[MH]*/*/*.png`

---

## Execution Workflow

The pipeline follows a sequential workflow. Execute notebooks/scripts in the order listed:

### Phase 1: Setup & Data Preparation (Steps 1-4)
1. **`0__setup.ipynb`** - Create conda/venv environments
2. **`1__download_template_for_registration__TemplateFlow.ipynb`** - Download MRI template
3. **`2__dataset_information.ipynb`** - Extract dataset metadata
4. **`3__Muslim_et_al_T2_orientation-pixdim_fix.ipynb`** - Fix image orientations

### Phase 2: Registration & Preprocessing (Steps 5-8)
5. **`4__rigid_registration_MI.py`** or **`4__rigid_registration_MI.ipynb`** - Rigid registration to template space
6. **`5__transform_lesion_masks_to_template_space.py`** or **`5__transform_lesion_masks_to_template_space.ipynb`** - Transform lesion masks
7. **`6__Muslim_et_al_post_registration_inspection.ipynb`** - Quality check registered scans
8. **`7__MRI_inspection__slice_range_selection.ipynb`** - Interactive slice range selection

### Phase 3: Data Augmentation & Training (Steps 9-10)
9. **`8__matlab_blur_and_contrast.ipynb`** or **`8__matlab_execute.py`** - Apply blur/contrast augmentations
10. **`8__train_using_isbi_test_so_have_train_GT_lesions_for_UQ__cleaned__out.ipynb`** - Train CNN model
    - **Output:** `best_model__baseline__20250503.keras`

### Phase 4: Prediction & Evaluation (Steps 11-12)
11. **`9-1-0__make_predictions_and_evaluate_model_3t.ipynb`** - Generate 3T predictions
    - **Output:** `all_unseen_3T_variant_scans_preds_for_baseline_model.pkl/csv`
12. **`9-1-1__make_predictions_and_evaluate_model_15t.ipynb`** - Generate 1.5T predictions
    - **Output:** `all_unseen_15T_variant_scans_preds_for_baseline_model.pkl/csv`

### Phase 5: Conformal Prediction Experiments (Step 13)
13. **`9-2-0__run_conformal_prediction_experiments.ipynb`** - Main conformal prediction experiments
    - **Inputs:** Prediction files from steps 11-12, `conformal.py`, `util.py`
    - **Outputs:** Multiple CP result files (PKL/CSV)

### Phase 6: Analysis (Steps 14-20)
14. **`9-2-1__marginal_coverage_vs_CC__3T_vs_15T__baseline__valid_cal_test.ipynb`** - Coverage analysis
15. **`9-2-2__visualize_two_example_slices_contrasting_marginal_vs_class-conditional.ipynb`** - Visualizations
16. **`9-2-3__conformal_efficiency.ipynb`** - Efficiency analysis
17. **`9-3__calculate_coverage_guarantee_statistics.ipynb`** - Coverage statistics
18. **`9-4-1__assess_coverage_guarantee_results__impact_of_calibration-test_distribution_differences_on_cp_coverage.ipynb`** - Distribution shift analysis
19. **`9-4-2__assess_coverage_guarantee_results__impact_of_calibration_set_size_on_empirical_coverage_distribution.ipynb`** - Calibration size analysis
20. **`9-4-3__assess_coverage_guarantee_results__impact_of_distribution_shift_on_prediction_set_size.ipynb`** - Prediction set size analysis

### Execution Chain Visualization
```
0__setup.ipynb
  └─> 1__download_template_for_registration__TemplateFlow.ipynb
      └─> 2__dataset_information.ipynb
          └─> 3__Muslim_et_al_T2_orientation-pixdim_fix.ipynb
              └─> 4__rigid_registration_MI.ipynb
                  └─> 5__transform_lesion_masks_to_template_space.ipynb
                      └─> 6__Muslim_et_al_post_registration_inspection.ipynb
                          └─> 7__MRI_inspection__slice_range_selection.ipynb
                              └─> 8__matlab_blur_and_contrast.ipynb
                                  └─> 8__train_using_isbi_test_so_have_train_GT_lesions_for_UQ__cleaned__out.ipynb
                                      └─> 9-1-0__make_predictions_and_evaluate_model_3t.ipynb
                                          └─> 9-1-1__make_predictions_and_evaluate_model_15t.ipynb
                                              └─> 9-2-0__run_conformal_prediction_experiments.ipynb
                                                  ├─> 9-2-1__marginal_coverage_vs_CC__3T_vs_15T__baseline__valid_cal_test.ipynb
                                                  ├─> 9-2-2__visualize_two_example_slices_contrasting_marginal_vs_class-conditional.ipynb
                                                  ├─> 9-2-3__conformal_efficiency.ipynb
                                                  ├─> 9-3__calculate_coverage_guarantee_statistics.ipynb
                                                  ├─> 9-4-1__assess_coverage_guarantee_results__impact_of_calibration-test_distribution_differences_on_cp_coverage.ipynb
                                                  ├─> 9-4-2__assess_coverage_guarantee_results__impact_of_calibration_set_size_on_empirical_coverage_distribution.ipynb
                                                  └─> 9-4-3__assess_coverage_guarantee_results__impact_of_distribution_shift_on_prediction_set_size.ipynb
```

---

## File Dependencies

### Critical Files Required for Execution

| File | Required By | Status |
|------|--------------|--------|
| `conformal.py` | `9-2-0__run_conformal_prediction_experiments.ipynb`, `9-1-0__make_predictions_and_evaluate_model_3t.ipynb`, `9-1-1__make_predictions_and_evaluate_model_15t.ipynb` | ✅ Present |
| `util.py` | `9-2-0__run_conformal_prediction_experiments.ipynb`, `9-1-0__make_predictions_and_evaluate_model_3t.ipynb`, `9-1-1__make_predictions_and_evaluate_model_15t.ipynb`, `8__train_using_isbi_test_so_have_train_GT_lesions_for_UQ__cleaned__out.ipynb` | ✅ Present |
| `best_model__baseline__20250503.keras` | `9-1-0__make_predictions_and_evaluate_model_3t.ipynb`, `9-1-1__make_predictions_and_evaluate_model_15t.ipynb` | ✅ Present |
| `all_unseen_3T_variant_scans_preds_for_baseline_model.pkl` | `9-2-0__run_conformal_prediction_experiments.ipynb` | ✅ Present |
| `all_unseen_15T_variant_scans_preds_for_baseline_model.pkl` | `9-2-0__run_conformal_prediction_experiments.ipynb` | ✅ Present |
| `matlab_functions/*.m` | `8__matlab_execute.py` | ✅ Present |
| `extra-20250407.csv` | `9-1-0__make_predictions_and_evaluate_model_3t.ipynb` | ✅ Present |

### Python Scripts Overview

| File | Purpose | Dependencies |
|------|---------|--------------|
| `3-1__reorient_and_apply_identity.py` | Interactive reorientation of MRI scans | External MRI data |
| `4__rigid_registration_MI.py` | Rigid registration to template space | External MRI data, TemplateFlow template |
| `5__transform_lesion_masks_to_template_space.py` | Transform lesion masks to template space | Registered scans, transform files |
| `7__post_registration_slice_range_selector.py` | Interactive slice range selection | Registered MRI scans |
| `8__matlab_execute.py` | Apply MATLAB blur/contrast operations | MATLAB, `matlab_functions/` directory |

### Key Data Files

| File | Used By | Purpose |
|------|---------|---------|
| `all_unseen_3T_variant_scans_preds_for_baseline_model.pkl/csv` | `9-2-0__run_conformal_prediction_experiments.ipynb` | 3T predictions |
| `all_unseen_15T_variant_scans_preds_for_baseline_model.pkl/csv` | `9-2-0__run_conformal_prediction_experiments.ipynb` | 1.5T predictions |
| `x4_cal-test_combos__100x_cp__per_variant_test_data__cp_instance_col.pkl` | Multiple analysis notebooks | Main CP results |
| `train.csv`, `val.csv` | Training notebook | Train/validation splits |

---

## External Dependencies

| Dependency | Used By | Notes |
|------------|---------|-------|
| MRI NIfTI files | Steps 1-7 | External data, paths: `./Muslim_et_al/`, `./ISBI/`, `./IXI/` |
| TemplateFlow template | Step 4 | Auto-downloaded via TemplateFlow |
| MATLAB R2023b | `8__matlab_execute.py` | Required for blur/contrast operations |
| ANTs | `4__rigid_registration_MI.py`, `5__transform_lesion_masks_to_template_space.py` | Required for registration |
| MRI slice PNGs | Steps 8-9 | External data, various variant folders |

---

## Notes

- **Large files:** The `.keras` model file may be large; consider using Git LFS if needed
- **Environment setup:** Detailed conda environment specifications are in `0__setup.ipynb`
- **Data paths:** Update paths in scripts/notebooks to match your local data directory structure
- **MATLAB:** Ensure MATLAB is properly configured with Python engine before running augmentation scripts

---

