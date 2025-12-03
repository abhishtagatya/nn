#!/bin/bash
## change this file to your needs

#echo "Adding some modules"

# module add gcc-10.2

start=$(date +%s)

echo "#################"
echo "    COMPILING    "
echo "#################"

## dont forget to use comiler optimizations (e.g. -O3 or -Ofast)
g++ -Wall -std=c++17 -O3 src/main.cpp src/*.hpp -Wno-sign-compare -Wno-unknown-pragmas -o network


echo "#################"
echo "     RUNNING     "
echo "#################"

epochs=30
learning_rate=0.001
batch_size=1024

## use nice to decrease priority in order to comply with aisa rules
## https://www.fi.muni.cz/tech/unix/computation.html.en
## especially if you are using multiple cores
nice -n 19 ./network $epochs $learning_rate $batch_size

end=$(date +%s)

echo "#################"
echo "     Finished    "
echo "#################"
echo "Total runtime: $((end - start)) seconds"
