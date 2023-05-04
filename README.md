Notes:
- The setup files should be sourced in a csh shell
- Even though snakemake is run in a csh shell, it still calls commands with bash. Note that in the cbig rule, the path expansions use $PWD which is valid bash but is not valid csh. 

TODO:
- Fix anat_d definitions, rollback changes to cbig repo
- Remove sorting from scripts and explicitly confirm that runs correspond in all files (slice timing)



# Tutorial

## Data Prep
```
# Uncompress files files
> mkdir /data/aint/RAW/00/pre/tar
> mkdir /data/aint/RAW/00/pre/untarred
> cd /data/aint/RAW/00/pre
> for f in tar/*.tar; do tar -xf $f -C untarred; done;

# Convert to BIDS format
> cd /data/aint/data 
> conda activate /data/nimlab/environment/conda/cbig
> cd /data/aint/cbig_workflow
> heudiconv -b -d /data/aint/RAW/{subject}/{session}/untarred/1/*/*/*/*.IMA -c dcm2niix -f scripts/heuristic.py -s 00 -ss pre -o /data/aint/cbig_workflow/data/BIDS 
```

## Run the pipeline!
```
> csh
> source source scripts/CBIG_setup_eristwo.csh
> source scripts/freesurfer_setup.csh
# The -np flag indicates that this is a dry run. Check that the pipeline steps look like what you want before proceeding
> snakemake -np data/connectivity/sub-00_ses-pre/sub-00_ses-pre_searchlight.nii.gz
```