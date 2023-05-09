import json
from datetime import datetime
from nimlab import connectomics as cs
from nimlab import functions as fn
from nilearn import image
from numpy.linalg import inv
import numpy as np
import os

# Run freesurfer recon-all, which is a prerequisite to CBIG
rule recon_all:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_T1w.nii.gz"
    output:
        directory("data/fs_subjects/sub-{sub}_ses-{ses}"),
        "data/fs_subjects/sub-{sub}_ses-{ses}/scripts/recon-all.done"
    run:
        # Snakemake auto-creates output dirs, but freesurfer fails if output dirs exist
        shell(f"rm -rf data/fs_subjects/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        # Don't destroy our node
        shell(f"nice -n 19 bash scripts/run_recon.sh data/fs_subjects {input} sub-{wildcards.sub}_ses-{wildcards.ses}")


# Prepare and format misc files needed by CBIG
rule create_fmrinii:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt"
    run:
        shell(f"mkdir -p data/cbig_configs/sub-{wildcards.sub}_ses-{wildcards.ses}")
        basepath = f"{input}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        #print(basepath)
        runs = glob_wildcards(basepath + "_task-rest_run-{run}_bold.nii.gz").run
        with open(output[0],"w+") as f:
            for r in runs:
                f.write(f"{r} {basepath}_task-rest_run-{r}_bold.nii.gz \n")

rule create_multiecho_fmrinii:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt"
    run:
        shell(f"mkdir -p data/cbig_configs/sub-{wildcards.sub}_ses-{wildcards.ses}")
        basepath = f"{input}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        #print(basepath)
        runs, echos = glob_wildcards(basepath + "_task-rest_run-{run}_echo-{echo}_bold.nii.gz")
        run_echos = {}
        for i in zip(runs, echos):
            if i[0] not in run_echos:
                run_echos[i[0]] = [i[1]]
                continue
            run_echos[i[0]].append(i[1])
        with open(output[0],"w+") as f:
            for r in run_echos:
                line = str(r)
                for e in run_echos[r]:
                    line += f" {basepath}_task-rest_run-{r}_echo-{e}_bold.nii.gz"
                f.write(line + "\n")

rule create_st_file:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    run:
        shell("python scripts/reformat_slicetiming.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}")

rule create_multiecho_st_files:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/"
    output:
        directory("data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings")
    run:
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        runs, echos = glob_wildcards(basepath + "_task-rest_run-{run}_echo-{echo}_bold.nii.gz")
        shell("mkdir -p {output[0]}")
        for e in set(echos):
            shell("python scripts/reformat_slicetiming_multiecho.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}/sub-{wildcards.sub}_ses-{wildcards.ses}_echo-{e}_slicetiming.txt -e {e}")

rule create_echotimes_file:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/"
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"

    run:
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        echos = glob_wildcards(basepath + "_task-rest_run-01_echo-{echo}_bold.nii.gz")[0]
        echotimes = []
        for e in echos:
            with open(basepath + f"_task-rest_run-01_echo-{e}_bold.json") as f:
                sidecar = json.load(f)
                echotimes.append(str(round(sidecar['EchoTime'] * 1000,2)))
        with open(output[0],"w+") as f:
            f.write(",".join(echotimes))

        
        

        
# Run CBIG
rule cbig:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    output:
        "cbig_sub-{sub}_ses-{ses}_singleecho_done.txt"
    run:
        begin_time = datetime.now()
        shell("mkdir -p data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        shell("export st_file=$PWD/{input[2]} && \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                nice -n 19 csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config $PWD/scripts/CBIG_preproc_fMRI_preprocess.config\"")
        end_time = datetime.now()
        begin_time_str = begin_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"cbig_{wildcards.sub}_{wildcards.ses}_done.txt", "w+") as f:
            f.write(f"CBIG started {begin_time_str} \n")
            f.write(f"CBIG ended {end_time_str} \n")
            f.write(f"CBIG took {(end_time - begin_time).total_seconds() / 60} minutes")

rule cbig_multiecho:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings",
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"
    output:
        "cbig_sub-{sub}_ses-{ses}_multiecho_done.txt",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz"
    run:
        sub = wildcards.sub
        ses = wildcards.ses
        begin_time = datetime.now()
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-AP_epi.json") as f:
            ap_trt = json.load(f)["TotalReadoutTime"]
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-PA_epi.json") as f:
            pa_trt = json.load(f)["TotalReadoutTime"]
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_run-01_echo-1_bold.json") as f:
            ees = json.load(f)["EffectiveEchoSpacing"] * 1000
        with open(f"data/BIDS/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-rest_run-01_echo-1_bold.json") as f:
            te = json.load(f)["EchoTime"] * 1000

        ap_path = os.path.abspath(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-AP_epi.nii.gz")
        pa_path = os.path.abspath(f"data/BIDS/sub-{sub}/ses-{ses}/fmap/sub-{sub}_ses-{ses}_dir-PA_epi.nii.gz")
        shell("mkdir -p data/cbig_output/sub-{sub}_ses-{ses}/")
        shell("export st_files=$(ls -p $PWD/{input[2]}/* | tr '\n' ',') && \
                export met_val=$(cat {input[3]}) && \
                \
                export j_minus_image_path={ap_path} && \
                export j_plus_image_path={pa_path} && \
                export j_minus_trt={ap_trt} && \
                export j_plus_trt={pa_trt} && \
                export ees={ees} && \
                export te={te} && \
                \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                nice -n 19 csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/BWH_multiecho.config\"")
        end_time = datetime.now()
        begin_time_str = begin_time.strftime("%d/%m/%Y %H:%M:%S")
        end_time_str = end_time.strftime("%d/%m/%Y %H:%M:%S")
        with open(f"cbig_sub-{wildcards.sub}_ses-{wildcards.ses}_multiecho_done.txt", "w+") as f:
            f.write(f"CBIG started {begin_time_str} \n")
            f.write(f"CBIG ended {end_time_str} \n")
            f.write(f"CBIG took {(end_time - begin_time).total_seconds() / 60} minutes")

# Concatenate final output timecourses
rule concat_tcs:
    input:
        "cbig_sub-{sub}_ses-{ses}_multiecho_done.txt",
        #"data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol"
    output:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz"
    run:
        shell(f"fslmerge -t {output[0]} data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/{wildcards.sub}/vol/*_finalmask.nii.gz")

# Split concatenated output in half for intra-subject qc purposes
rule split_concat:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz"
    output:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz",
    run:
        concat_img = image.load_img(input[0])
        half = int(concat_img.shape[3] / 2)
        split1 = image.index_img(concat_img, slice(0, half))
        split2 = image.index_img(concat_img, slice(half,None))
        split1.to_filename(output[0])
        split2.to_filename(output[1])


# Calculate connectivity to depression map
rule depression_conn:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz",
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_AllFX_wmean_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_AllFX_wmean_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_AllFX_wmean_z.nii.gz"
    run:
        res = cs.singlesubject_seed_conn("standard_data/AllFX_wmean.nii.gz", input[0], transform="zscore")
        res.to_filename(output[0])
        res = cs.singlesubject_seed_conn("standard_data/AllFX_wmean.nii.gz", input[1], transform="zscore")
        res.to_filename(output[1])
        res = cs.singlesubject_seed_conn("standard_data/AllFX_wmean.nii.gz", input[2], transform="zscore")
        res.to_filename(output[2])

rule searchlight:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz",
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_peak.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_peak.txt",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_peak.txt"
    run:
        shell(f"python scripts/searchlight.py {input[0]} {output[0]} {output[1]}")
        shell(f"python scripts/searchlight.py {input[1]} {output[2]} {output[3]}")
        shell(f"python scripts/searchlight.py {input[2]} {output[4]} {output[5]}")

rule searchlight_spatialcorr:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz",
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_spatialcorr.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_searchlight_spatialcorr.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_searchlight_spatialcorr.nii.gz"
    run:
        shell(f"python scripts/searchlight_spatialcorr.py {input[0]} {output[0]}")
        shell(f"python scripts/searchlight_spatialcorr.py {input[1]} {output[1]}")
        shell(f"python scripts/searchlight_spatialcorr.py {input[2]} {output[2]}")

rule cluster_cog:
    input:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_AllFX_wmean_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_AllFX_wmean_z.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_AllFX_wmean_z.nii.gz",
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_cog.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_cog.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_cog.nii.gz"
    run:
        shell(f"python scripts/cluster_cog.py {input[0]} {output[0]}")
        shell(f"python scripts/cluster_cog.py {input[1]} {output[1]}")
        shell(f"python scripts/cluster_cog.py {input[2]} {output[2]}")


rule subject_target:
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/norm_MNI152_1mm.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_searchlight_peak.txt"
    output:
        "data/target/sub-{sub}_ses-{ses}/sub_target.txt"
    run:
        #os.makedirs(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/",exist_ok=True)
        def center_of_gravity(arr):
            x = 0
            y = 0
            z = 0
            total = 0
            for c in np.ndindex(arr.shape):
                if arr[c] == 0:
                    continue
                total += arr[c]
                x += c[0] * arr[c]
                y += c[1] * arr[c]
                z += c[2] * arr[c]
            return x / total, y / total, z / total

        ref = image.load_img(input[1])
        with open(input[2]) as f:
            lines = [l.strip() for l in f.readlines()]
        coord_strings = lines[1].split(": ")[-1].replace("(","").replace(")","").split(",")
        print(coord_strings)
        coords = (float(coord_strings[0]), float(coord_strings[1]), float(coord_strings[2])) 
        print(coords)

        blank = np.zeros(ref.shape)
        vox_coords = image.coord_transform(coords[0], coords[1], coords[2], inv(ref.affine))
        mni_sphere = fn.make_sphere(ref, vox_coords, 2, 100)
        mni_sphere.to_filename(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/mni.nii.gz")

        # Transform to FS space
        shell(f"SUBJECTS_DIR=/data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/ \
                mri_vol2vol --mov data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/mni.nii.gz\
                --targ /data/nimlab/software/CBIG_nimlab/CBIG/data/templates/volume/FSL_MNI152_FS4.5.0/mri/norm.nii.gz \
                --s FSL_MNI152_FS4.5.0 --m3z talairach.m3z --o data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs.nii.gz --no-save-reg --interp cubic")

        # Threshold out interpolation noise
        image.threshold_img(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs.nii.gz",10) \
            .to_filename(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs_thresh.nii.gz")
        fs_thresh_fullpath = os.path.abspath(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/fs_thresh.nii.gz")
        sub_target_fullpath = os.path.abspath(f"data/target/sub-{wildcards.sub}_ses-{wildcards.ses}/sub.nii.gz")
        shell(f"cd {input[0]}/mri/ && \
                export SUBJECTS_DIR={os.path.abspath('data/fs_subjects')} && \
                mri_vol2vol --mov norm.mgz --s sub-{wildcards.sub}_ses-{wildcards.ses} \
                --targ {fs_thresh_fullpath}  --o {sub_target_fullpath} --no-save-reg --interp cubic --inv-morph")

        sub_transformed = image.threshold_img(sub_target_fullpath,10)
        sub_target_vox = center_of_gravity(sub_transformed.get_fdata())
        sub_target_coord = image.coord_transform(sub_target_vox[0], sub_target_vox[1], sub_target_vox[2], sub_transformed.affine)
        with open(output[0], "w+") as f:
            f.write(f"{sub_target_coord[0]}, {sub_target_coord[1]}, {sub_target_coord[2]}\n")


# QC metrics
rule mriqc:
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/"
    output:
        directory("data/qc/mriqc/sub-{sub}/ses-{ses}/"),
    run:
        shell("mkdir -p data/qc/mriqc/")
        shell("sh scripts/qc/run_mriqc.sh data/BIDS data/qc/mriqc/ {wildcards.sub}")

rule window_clust:
    input:
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_concat.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-1.nii.gz",
        "data/cbig_output/sub-{sub}_ses-{ses}/{sub}/vol/sub-{sub}_ses-{ses}_split-2.nii.gz",
    output:
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_windowclust.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-1_windowclust.nii.gz",
        "data/connectivity/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_split-2_windowclust.nii.gz"
    run:
        shell(f"python scripts/window_clust.py {input[0]} {output[0]}")
        shell(f"python scripts/window_clust.py {input[1]} {output[1]}")
        shell(f"python scripts/window_clust.py {input[2]} {output[2]}")