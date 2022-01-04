## the master-script
As mentioned before, the master script controls the pipeline and consists of "modules" prefixed by `spinoza_` ranging from 00-20 that all perform a specific task. You can run a certain module by setting the `-m` flag that is to follow the master script. It is important that this script is ran from the directory it is located in because it needs to find the setup script to load all paths. The entire pipeline assumes that you have the data in BIDS format; if not, bad things will happen and it will not be able to find the required files. So please make sure it is roughly in BIDS format, with specifically the `sub-xxx`, `ses-xxx`, and suffixes set properly.

So, let's say I want to run the 9th module, I can type in the command line:
```
master -m 09
```

If you want to run multiple modules successively, you can specify the desired modules in a comma-separated fashion:
```
master -m 01,02,10,11
```
This will run module 1,2,10, and 11.

Some modules have been clustered already and the clusters of modules can be found at the top of the master script, or by typing `./master` which will give you the usage-info. For instance, if you have MP2RAGE and MEMP2RAGE data, you can have all preprocessing steps such as registration and averaging done by the cluster `PRE` by typing:
```
master -m PRE
```

Tip: it could be useful to make an alias in your ~/.bash_profile file to change directory to the programs:
```
gedit ~/.bash_profile
alias PROG="cd /path/to/masterscript"
```

You can then type `PROG` in the terminal to change directory to the master script.

## the setup-script
All scripts controlled by the master-script call upon `spinoza_setup`, a setup file containing variables and paths. Should you want to adapt parts of this pipeline for your own purposes, you might want to change the `PATH_HOME` variable. All other paths use this variable, so changing this one variable will adapt the pipeline to your system. The anatomical part of this pipeline is built to process either MP2RAGE data only, or a combination of MP2RAGE and multi-echo (ME-) MP2RAGE data. This switch can also be set in the setup script with the variable `space`. If set to `mp2rage`, the pipeline will only process MP2RAGE files, while if set to `average`, it will first register the MEMP2RAGE to the MP2RAGE and then calculate an average, resulting in outputs in "average space".