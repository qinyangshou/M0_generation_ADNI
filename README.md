# This project is developed by Qinyang Shou at University of Southern California (USC), under Lab of Functional MRI Technology (LOFT, www.loft-lab.org)

# Work can be cited as:
Shou, Q., Cen, S., Chen, N., Ringman, J. M., Kim, H., Jack, C. R., Borowski, B., Senjem, M. L., Arani, A., Wang, D. J. J., & for the Alzheimer’s Disease Neuroimaging Initiative. (under revision). Generative diffusion model enables quantification of calibration-free arterial spin labeling perfusion MRI data in an Alzheimer’s Disease cohort.

Model weights can be requested from the institute upon publication.

# To set up the environment:
conda env create -f environment.yaml
conda activate ldm

# To run the inference on ADNI dataset:
python sample_M0_LDM_ADNI.py --indir ./input_data --outdir ./generated_data --steps 50 --num_averages 20 --model_name 2025-02-14T12-08-21_finetune_Mayo_combined

# Acknowledgement:
The code is branched from Latent Diffusion Models(https://github.com/CompVis/latent-diffusion)
