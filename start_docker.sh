sudo usermod -aG docker $USER
newgrp docker
docker ps

./turboprep-docker images_registered/Patient-001_week-000-2_reg/Patient-001_week-000-2_reg_0003.nii.gz Turbo_prep_processed_FLAIR/Patient-001_week-000-2_reg/ MNI152_T1_1mm_brain.nii.gz --modality flair