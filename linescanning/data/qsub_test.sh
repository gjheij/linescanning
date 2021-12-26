#!/bin/bash
#$ -S /bin/bash
#$ -N qsub_test
#$ -j Y
#$ -q veryshort.q
#$ -V

echo "sleeping..."
sleep 5