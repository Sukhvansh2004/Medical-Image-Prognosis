# Medical Image Prognosis Preprocessing & Registration

This repository is part of the **Medical Prognosis** project under the EE-608 course at IIT Mandi. It implements the **preprocessing** and **registration** stages of the pipeline, preparing brain MRI scans (NIfTI) for downstream deep-learning–based prognosis.

## Overview

The pipeline performs the following high-level steps:

1. **File and Directory Validation**  
   - Ensures the input dataset directory structure is correct via `check.py`.  
2. **Data Conversion**  
   - Generates standardized lists of input and output file paths using `convert.py`.  
3. **Image Enhancement & Normalization**  
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE), MSRCR, and white‑stripe normalization with `normalize.py` and `normalize2.py`.  
   - `msrcr.py` implements the Multi‑Scale Retinex with Color Restoration algorithm and demonstrates resampling to required aspect ratios before enhancement.  
4. **Masking & Skull‑Stripping**  
   - Generates or applies brain masks via `mask.py` for region‑of‑interest extraction.  
5. **Registration to MNI Template**  
   - Aligns scans to the **MNI152** template (`MNI152_T1_1mm_brain.nii.gz`) using the integrated [TurboPrep](https://github.com/LemuelPuglisi/turboprep) toolkit.  
   - Note: TurboPrep's segmentation/registration may require additional configuration—see `turboprep_processing_log.txt` for details.  
6. **Segmentation & Volume Computation**  
   - Performs MRI segmentation using SynthSeg via `segment.py`.  
   - Computes region volumes from segmented labels in `volumes_process.py`.  
7. **Batch Orchestration**  
   - Runs the full CPU pipeline with `script.py` or the GPU‑accelerated version with `script_gpu.py`.  

## File Structure

```
.
├── BrLP/                         # Prognosis diffusion model (see link below)
├── turboprep/                    # Integration of TurboPrep toolkit
│   ├── start_docker.sh           # Launches Docker for TurboPrep
│   └── turboprep_processing_log.txt
├── MNI152_T1_1mm_brain.nii.gz    # Standard MNI152 template
├── check.py                      # Validates dataset structure
├── convert.py                    # Prepares input/output path lists
├── mask.py                       # Brain masking / skull-stripping
├── msrcr.py                      # MSRCR enhancement implementation
├── msrcr_sample.py               # Resamples and applies MSRCR
├── normalize.py                  # CLAHE, MSRCR, white-stripe normalization
├── normalize2.py                 # White-stripe normalization only
├── refine.py                     # Checks input/output correspondence
├── reg_process_0000.py           # Registration pipeline (TurboPrep)
├── segment.py                    # MRI segmentation (SynthSeg)
├── volumes_process.py            # Volume calculation from labels
├── script.py                     # CPU batch orchestrator
├── script_gpu.py                 # GPU-accelerated orchestration
├── sample.py                     # NIfTI sampling utilities
├── sorter.ipynb                  # File sorting and inspection notebook
├── input_files.txt               # Batch input file list
├── input_files_2.txt             # Alternative input list
├── data_formated.csv             # Scan metadata / parameters
├── output_paths.txt              # Batch output paths
├── output_paths_2.txt            # Alternative output paths
├── outputs/                      # Example processed outputs
├── outputs.nii                   # Sample single-output scan
├── tree                          # Utility to display dir tree
└── README.md                     # This file
```

## External References

- **BrLP**: Prognosis diffusion model repository used for downstream prognosis  
  [https://github.com/Sukhvansh2004/BrLP](https://github.com/Sukhvansh2004/BrLP)
- **TurboPrep**: Image segmentation and registration toolkit  
  [https://github.com/LemuelPuglisi/turboprep](https://github.com/LemuelPuglisi/turboprep)

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/Sukhvansh2004/Medical-Image-Prognosis.git
   cd Medical-Image-Prognosis
   ```

2. **(Optional) Clone submodules**  
   ```bash
   git clone https://github.com/Sukhvansh2004/BrLP.git BrLP
   git clone https://github.com/LemuelPuglisi/turboprep.git turboprep
   ```

3. **Set up Python environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install numpy scipy nibabel SimpleITK pandas scikit-image torch
   ```

4. **Run the pipeline**  
   - **CPU mode**:  
     ```bash
     python script.py --inputs input_files.txt --outputs output_paths.txt --template MNI152_T1_1mm_brain.nii.gz --csv data_formated.csv
     ```  
   - **GPU mode**:  
     ```bash
     python script_gpu.py --inputs input_files.txt --outputs output_paths.txt --template MNI152_T1_1mm_brain.nii.gz --csv data_formated.csv
     ```

5. **(Optional) Docker for TurboPrep**  
   ```bash
   cd turboprep
   ./start_docker.sh
   ```

## Usage Notes

- Ensure **SynthSeg** is installed for `segment.py` (see its [GitHub page](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg)).
- Review `turboprep_processing_log.txt` for any errors in the TurboPrep stage.
- Use `sorter.ipynb` to visually inspect and sort processed outputs.

## License

This project is licensed under the MIT License. See `LICENSE.md` for details.
