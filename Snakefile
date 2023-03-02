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

rule cbig:
    output:
        "cbig_{sub}_{ses}_done.txt"
    input:
        directory("data/fs_subjects/sub-{sub}_ses-{ses}")
    run:
        #shell("csh -c \"source scripts/CBIG_config_erstwo.csh\"")
        shell("csh -c \"source scripts/CBIG_config_erstwo.csh && $CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/stable_projects/preprocessing/CBIG_fMRI_Preproc2016\"")
