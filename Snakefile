from datetime import datetime

rule all:
    output:
        "finished.txt"
    shell:
        "touch finished.txt"

# Run freesurfer recon-all, which is a prerequisite to CBIG
rule recon_all:
    output:
        directory("data/fs_subjects/sub-{sub}_ses-{ses}")
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/anat/sub-{sub}_ses-{ses}_e2_T1w.nii.gz"
    run:
        shell("mkdir -p data/fs_subjects")
        shell(f"scripts/run_recon.sh data/fs_subjects {input} sub-{wildcards.sub}_ses-{wildcards.ses}")



# Prepare and format misc files needed by CBIG
rule create_fmrinii:
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt"
    input:
        "data/BIDS/sub-{sub}/ses-{ses}",
    run:
        shell(f"mkdir -p data/cbig_configs/sub-{wildcards.sub}_ses-{wildcards.ses}")
        basepath = f"{input}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        #print(basepath)
        runs = glob_wildcards(basepath + "_task-rest_run-{run}_bold.nii.gz").run
        with open(output[0],"w+") as f:
            for r in runs:
                f.write(f"{r} {basepath}_task-rest_run-{r}_bold.nii.gz \n")

rule create_st_file:
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    input:
        "data/BIDS/sub-{sub}/ses-{ses}",
    run:
        shell("python scripts/reformat_slicetiming.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}")
        

# Run CBIG
rule cbig:
    output:
        "cbig_{sub}_{ses}_done.txt"
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    run:
        begin_time = datetime.now()
        shell("mkdir -p data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        shell("export st_file=$PWD/{input[2]} && \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
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
