#!/bin/bash
if [ -z "$11" ]
then
TEST_FAPATH=$1
DBPATH=$2
HHLIB=$3
CONFIG=$4
CCO=$5
MFO=$6
BPO=$7
DICT=$8
RES=$9
DEVICE=$10
else
CONDA=$1
ENV=$2
TEST_FAPATH=$3
DBPATH=$4
HHLIB=$5
CONFIG=$6
CCO=$7
MFO=$8
BPO=$9
DICT=${10}
RES=${11}
DEVICE=${12}
SOURCE_PATH=${13}
source ${CONDA}/bin/activate
conda activate "${ENV}"
fi

cd $SOURCE_PATH
python scripts/predict.py \
--gpu-ids $DEVICE \
$TEST_FAPATH \
$DBPATH \
$HHLIB \
$CONFIG \
$CCO \
$MFO \
$BPO \
$DICT \
$RES



