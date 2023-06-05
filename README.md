Notes:
- The setup files should be sourced in a csh shell
- Even though snakemake is run in a csh shell, it still calls commands with bash. Note that in the cbig rule, the path expansions use $PWD which is valid bash but is not valid csh. 

TODO:
- Fix anat_d definitions, rollback changes to cbig repo
- Remove sorting from scripts and explicitly confirm that runs correspond in all files (slice timing)



# Tutorial
## Start a tmux session
```
# Log into our node
> ssh cl20@sna003a.research.partners.org
# If you have no existing session, run
> tmux
# Else, attach to your existing session
> tmux ls
> tmux attach -t <your session number>
```

## Data Prep
```
# Create new directories
> mkdir /data/aint/RAW/00/pre/tar
> mkdir /data/aint/RAW/00/pre/untarred

# Copy over files from Lisa server (commands omitted)

# Uncompress files
> cd /data/aint/RAW/00/pre
> for f in tar/*.tar; do tar -xf $f -C untarred; done;

# Convert to BIDS format
> conda activate /data/nimlab/environment/conda/cbig
> cd /data/aint/cbig_workflow
> heudiconv -b -d /data/aint/RAW/{subject}/{session}/untarred/1/*/*/*/*.IMA -c dcm2niix -f scripts/heuristic.py -s 00 -ss pre -o /data/aint/cbig_workflow/data/BIDS 
```

## Run the pipeline!
The pipeline is coordinated using [Snakemake](https://github.com/snakemake/snakemake) which automatically calculates what steps need to be run in order to produce a desired file. Since we want to produce the target coordinates, we tell Snakemake to produce `data/target/sub-00_ses-pre/sub_target.txt`
```
> csh
> source scripts/CBIG_setup_eristwo.csh
> source scripts/freesurfer_setup.csh
# The -np flag indicates that this is a dry run. Check that the pipeline steps look like what you want before proceeding
> snakemake -np data/target/sub-00_ses-pre/sub_target.txt
# If everything seems fine, run it with c1 to specify 1 parallel job. Using more is possible if running multiple subjects, but may overload the node.
> snakemake -pc1 data/target/sub-00_ses-pre/sub_target.txt data/qc/sub-00_ses-pre/seed_corrs.txt
```
