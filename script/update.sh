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
MODE=$1
HOME=/root/ElasticCDC
CONF_PATH=$HOME/conf
BIN_PATH=$HOME/build/install/bin

if [ $MODE == "config" ]; then
{
    for ((i=0;i<${#IP_LIST[@]};i++));
    do
    {
        IP=${IP_LIST[i]}
        scp $CONF_PATH/*.json $USER@$IP:$CONF_PATH/
    } &
    done
    wait
}
elif [ $MODE == "program" ]; then
{
    for ((i=0;i<${#IP_LIST[@]};i++));
    do
    {
        IP=${IP_LIST[i]}
        scp $BIN_PATH/* $USER@$IP:$BIN_PATH/
    } &
    done
    wait
}
else
{
    for ((i=0;i<${#IP_LIST[@]};i++));
    do
    {
        IP=${IP_LIST[i]}
        scp $CONF_PATH/*.json $USER@$IP:$CONF_PATH/
        scp $BIN_PATH/* $USER@$IP:$BIN_PATH/
    } &
    done
    wait
}
fi
