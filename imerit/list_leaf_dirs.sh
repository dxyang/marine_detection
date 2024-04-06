#! /bin/bash
# Usage: ./list_leaf_dirs <top-level directory>
# Source: https://stackoverflow.com/questions/1574403/list-all-leaf-subdirectories-in-linux
find $1 -type d | sort | awk '$0 !~ last "/" {print last} {last=$0} END {print last}'
