rule all:
    output:
        "finished.txt"
    shell:
        "touch finished.txt"

rule recon_all:
    output:
        directory("data/fs_subjects/sub-{sub}_ses-{ses}")
    input:
        "data/BIDS/sub-{sub}/ses-pre/anat/sub-{sub}_ses-{ses}_run-1_T1w.nii.gz"
    run:
        shell("mkdir -p data/fs_subjects")
        shell(f"scripts/run_recon.sh data/fs_subjects {input} sub-{wildcards.sub}_ses-{wildcards.ses}")

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
        


rule cbig:
    output:
        "cbig_{sub}_{ses}_done.txt"
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_fmrinii.txt"
    run:
        #shell("csh -c \"source scripts/CBIG_config_erstwo.csh\"")
        shell("mkdir -p data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        shell("SUBJECTS_DIR={input[0]} csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii {input[1]} \
                -anat_s {input[0]} \
                -output_d data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/preproc_CBIG_Butler.config\"")
