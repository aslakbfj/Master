#!/bin/bash



# download metrics from the cluster
DIR="C:/Users/aslak/Master/Github_first/SCRATCH/performance-metrics/mnt/SCRATCH/asfj/AS-TAC/ExplaiNN"
#echo "plot the metrics"
# Plot the performance metrics
#PY_SCRIPT=../../scripts/plot_metrics_local.py
#${PY_SCRIPT} -i metrics.csv -o ./optimize_units_metrics.png


# Initialize an empty string for the final output
output=""
# create the header which is "num_units", then all lines of ../../dna-sequence-models/downloads/bed_list_test.txt
#output+="num_cnns\tmetric\t$(awk 'BEGIN{ORS="\t"}{print $0}' ../../dna-sequence-models/downloads/bed_list_test.txt)\n"


echo -e "num_cnns\tmetric\t$(cat ../../dna-sequence-models/downloads/bed_list_test.txt | tr '\n' '\t')" > combined_performance_metrics.tsv
# For each directory in the optimized_units directory
  # For each directory
for dir in $(ls -d ${DIR}/*)
    do
    # If performance_metrics.tsv exists in the directory
    #echo $dir
        if [[ -f "${dir}/performance-metrics.tsv" ]]; then
        # Extract the last folder name
        last_folder_name=$(basename $dir)
        #remove "num_cnns_" from the folder name
        last_folder_name=${last_folder_name#num_cnns_}
        # Extract the second and third line and append it to the output string
       # output+="$last_folder_name\t$(sed -n '2p' ${dir}/performance-metrics.tsv)\n"
        output+="$last_folder_name\t$(sed -n '2p' ${dir}/performance-metrics.tsv | tr ' ' '\t')\n"
        # Extract the third line and append it to the output string
        #output+="$line$last_folder_name\t$(sed -n '3p' ${dir}/performance-metrics.tsv)\n"
        output+="$last_folder_name\t$(sed -n '3p' ${dir}/performance-metrics.tsv | tr ' ' '\t')\n"
    fi
done

# Print the output string
echo -e $output >> combined_performance_metrics.tsv
