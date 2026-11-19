#!/bin/sh

# Create NSYS report directory
mkdir -p $(pwd)/nsys_reports/

touch $(pwd)/nsys_reports/.gitkeep

# Run the profiler with the specified options

nsys profile \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --force-overwrite=true \
    --stats=false \
    --output="$(pwd)/nsys_reports/sparse-bregman" \
    --sample=none \
    --backtrace=none \
    --pytorch=autograd-nvtx \
    python3 src/main.py


# Calcuate number of reports already generated
NUM_REPORTS=$(ls -1 $(pwd)/nsys_reports/ | wc -l)
ID=$($RANDOM | md5sum | head -c 20)

# Create a new directory for the current report and move the generated report files into it
mkdir -p $(pwd)/nsys_reports/report-$NUM_REPORTS
mv $(pwd)/nsys_reports/sparse-bregman.nsys-rep $(pwd)/nsys_reports/report-$NUM_REPORTS/sparse-bregman-report.nsys-rep
# mv ~/nsys_profiles/report.sqlite ~/nsys_profiles/sparse-bregman/report-$NUM_REPORTS-$ID/report.sqlite
