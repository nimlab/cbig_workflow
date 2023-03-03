Notes:
- The setup files should be sourced in a csh shell
- Even though snakemake is run in a csh shell, it still calls commands with bash. Note that in the cbig rule, the path expansions use $PWD which is valid bash but is not valid csh. 

TODO:
- Fix anat_d definitions, rollback changes to cbig repo
- Remove sorting from scripts and explicitly confirm that runs correspond in all files (slice timing)