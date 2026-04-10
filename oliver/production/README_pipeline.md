**RE-RUNNING DATA FILES SCRIPTS**
When new data (in GCP bucket) is gathered and needs to be appended, the following pipeline needs to be run:

1. Overwrite the existing file with the new/exstended file in `data/gcp_manual_copy`
2. run file: **`oliver/production/appending_grades.ipynb`** to append all grades to the data
3. run file **`oliver/production/universal_load.ipynb`** to overwrite and store all variations of the final data file.
(will be stored in location: `data/data_analysis_files`) 

NOW ALL DATA FILES HAVE BEEN UPDATED!

**RE-RUNNING ANALYSIS SCRIPTS**
If there is a need to update analysis, run the following files which are final analysis scripts (and check if plots/text still allign with analysis with new data)
- run file **`oliver/production/2023_trends_analysis.ipynb`** and see plots from that file in location: `oliver/exported_plots/2023_trends_analysis`
- run file **`oliver/basic_stats_analysis.ipynb`** and see plot from that file in location: `oliver/exported_plots/basic_stats`