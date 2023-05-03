from nimlab import datasets as ds
from nimlab import functions as fn
import numpy as np
from nilearn import image
from numpy.linalg import inv
import os

ref = image.load_img("/data/nimlab/USERS/cl20/cbig_workflow/data/cbig_output/sub-3015_ses-01/3015/vol/norm_MNI152_1mm.nii.gz")
coords = (-44.0, 40.0, 32.0) 

blank = np.zeros(ref.shape)
vox_coords = image.coord_transform(coords[0], coords[1], coords[2], inv(ref.affine))
mni_sphere = fn.make_sphere(ref, vox_coords, 2, 100)
mni_sphere.to_filename("mni_tmp.nii.gz")

# os.system("SUBJECTS_DIR=/data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/ \
#           mri_vol2vol --mov data/cbig_output/sub-3015_ses-01/3015/vol/norm_MNI152_1mm.nii.gz\
#           --targ /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/FSL_MNI152_FS4.5.0/mri/norm.nii.gz \
#           --s FSL_MNI152_FS4.5.0 --m3z talairach.m3z --o scripttest1.nii.gz --interp cubic")

# Transform to FS space
os.system("SUBJECTS_DIR=/data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/ \
          mri_vol2vol --mov mni_tmp.nii.gz\
          --targ /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/FSL_MNI152_FS4.5.0/mri/norm.nii.gz \
          --s FSL_MNI152_FS4.5.0 --m3z talairach.m3z --o fs_tmp.nii.gz --no-save-reg --interp cubic")

# Threshold out interpolation noise
image.threshold_img("fs_tmp.nii.gz",10).to_filename("fs_thresh.nii.gz")
os.system("cd /data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects/sub-3015_ses-01/mri/ && \
          export SUBJECTS_DIR=/data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects && \
          mri_vol2vol --mov /data/nimlab/USERS/cl20/cbig_workflow/data/fs_subjects/sub-3015_ses-01/mri/norm.mgz --s sub-3015_ses-01 --targ /data/nimlab/USERS/cl20/cbig_workflow/fs_thresh.nii.gz  --o /data/nimlab/USERS/cl20/cbig_workflow/final.nii.gz --no-save-reg --interp cubic --inv-morph")