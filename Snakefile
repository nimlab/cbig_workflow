import json

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

rule create_multiecho_fmrinii:
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt"
    input:
        "data/BIDS/sub-{sub}/ses-{ses}",
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
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_slicetiming.txt"
    input:
        "data/BIDS/sub-{sub}/ses-{ses}",
    run:
        shell("python scripts/reformat_slicetiming.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}")

rule create_multiecho_st_files:
    output:
        directory("data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings")
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/",
    run:
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        runs, echos = glob_wildcards(basepath + "_task-rest_run-{run}_echo-{echo}_bold.nii.gz")
        shell("mkdir -p {output[0]}")
        for e in set(echos):
            shell("python scripts/reformat_slicetiming_multiecho.py -d data/BIDS -p sub-{wildcards.sub} -s ses-{wildcards.ses} -o {output[0]}/sub-{wildcards.sub}_ses-{wildcards.ses}_echo-{e}_slicetiming.txt -e {e}")

rule create_echotimes_file:
    output:
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"
    input:
        "data/BIDS/sub-{sub}/ses-{ses}/"
    run:
        basepath = f"{input[0]}/func/sub-{wildcards.sub}_ses-{wildcards.ses}"
        echos = glob_wildcards(basepath + "_task-rest_run-01_echo-{echo}_bold.nii.gz")[0]
        echotimes = []
        for e in echos:
            with open(basepath + f"_task-rest_run-01_echo-{e}_bold.json") as f:
                sidecar = json.load(f)
                echotimes.append(round(sidecar['EchoTime'] * 1000,2))
        with open(output[0],"w+") as f:
            for t in echotimes:
                f.write(str(t) + " ")

        
        

        
# Run CBIG
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

rule cbig_multiecho:
    output:
        "cbig_{sub}_{ses}_multiecho_done.txt"
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings",
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"
    run:
        begin_time = datetime.now()
        shell("mkdir -p data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        shell("export st_file=$PWD/{input[2]} && \
                export met_val=$(cat {input[3]}) && \
                source scripts/freesurfer_setup.bash && \
                echo $FREESURFER_HOME && \
                csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/preproc_CBIG_Butler.config\"")

rule cbig_multiecho:
    output:
        "cbig_{sub}_{ses}_multiecho_done.txt"
    input:
        "data/fs_subjects/sub-{sub}_ses-{ses}",
        "data/cbig_configs/sub-{sub}_ses-{ses}/sub-{sub}_ses-{ses}_multiecho_fmrinii.txt",
        "data/cbig_configs/sub-{sub}_ses-{ses}/multiecho_slicetimings",
        "data/cbig_configs/sub-{sub}_ses-{ses}/echotimes.txt"
    run:
        #shell("csh -c \"source scripts/CBIG_config_erstwo.csh\"")
        shell("mkdir -p data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses}/")
        #shell("export st_file={input[2]}")
        shell("export st_file=$PWD/{input[2]} && \
                export met_val=$(cat {input[3]}) && \
                source scripts/freesurfer_setup.bash && \
                csh -c \"$CBIG_CODE_DIR/stable_projects/preprocessing/CBIG_fMRI_Preproc2016/CBIG_preproc_fMRI_preprocess.csh \
                -s {wildcards.sub} \
                -fmrinii $PWD/{input[1]} \
                -anat_s sub-{wildcards.sub}_ses-{wildcards.ses} \
                -anat_d $PWD/data/fs_subjects \
                -output_d $PWD/data/cbig_output/sub-{wildcards.sub}_ses-{wildcards.ses} \
                -config scripts/BWH_multiecho.config\"")