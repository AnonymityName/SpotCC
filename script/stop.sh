#!/bin/bash

NODE_NUM=12
IP_LIST=(
    172.24.1.3
    172.24.1.188
    172.24.1.189
    172.24.1.190
    172.24.1.191
    172.24.1.192
    172.24.1.193
    172.24.1.194
    172.24.1.195
    172.24.1.196
    172.24.1.197
    172.24.1.198
)
USER=root
HOME=/root/ElasticCDC
BUILD_PATH=$HOME/build
BIN_PATH=$BUILD_PATH/install/bin

CONF_PATH=$HOME/conf
CONF_FILE=elasticcdc_test.json


BACK_BIN=backend_server
FRONT_BIN=image_frontend
CLIENT_BIN=image_client

DATASET=cifar10
INPUT_PATH=/$USER/input/$DATASET/data

# stop backend
for((i=0;i<$NODE_NUM;i++));
do
{
    IP=${IP_LIST[i]}
    ssh $USER@$IP "killall $BACK_BIN"
} &
done

# stop frontend
killall $FRONT_BIN

