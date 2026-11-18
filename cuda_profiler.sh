#!/bin/sh

# Create NSYS report directory
mkdir -p ~/nsys_profiles/sparse-bregman

# Run the profiler with the specified options
nsys profile --capture-range=cudaProfilerApi \
    --trace='cuda,cudnn,cublas,osrt,opengl,nvtx' \
    --force-overwrite=true --stats=false\
    --output=~/nsys_profiles/sparse-bregman \
    --wait=primary --sample=none --stop-on-exit=true\
    --delay=10 --pytorch=autograd-shapes-nvtx \
    --backtrace=none python3 src/main.py
    # --python-sampling=true --duration=10 --backtrace=none python3 src/main.py


# Calcuate number of reports already generated
NUM_REPORTS=$(ls -1 ~/nsys_profiles/sparse-bregman | wc -l)
ID=$($RANDOM | md5sum | head -c 20)

# Create a new directory for the current report and move the generated report files into it
mkdir -p ~/nsys_profiles/sparse-bregman/report-$NUM_REPORTS
rm -rf ~/nsys_profiles/sparse-bregman/*.Identifier
mv ~/nsys_profiles/sparse-bregman.nsys-rep ~/nsys_profiles/sparse-bregman/report-$NUM_REPORTS/sparse-bregman-report.nsys-rep
# mv ~/nsys_profiles/report.sqlite ~/nsys_profiles/sparse-bregman/report-$NUM_REPORTS-$ID/report.sqlite
