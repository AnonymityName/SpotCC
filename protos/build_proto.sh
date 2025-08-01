#!/bin/bash

~/ElasticCDC/build/third-party/protobuf/bin/protoc --cpp_out=./ *.proto

GRPC_CPP_PLUGIN=grpc_cpp_plugin
GRPC_CPP_PLUGIN_PATH=`which ${GRPC_CPP_PLUGIN}`

~/ElasticCDC/build/third-party/protobuf/bin/protoc --grpc_out=./ --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN_PATH} *.proto 
# ~/ElasticCDC/build/third-party/protobuf/bin/protoc --grpc_out=./ *.proto

mv *.grpc.* ~/ElasticCDC/src/protocol/
mv *.pb.* ~/ElasticCDC/src/protocol/