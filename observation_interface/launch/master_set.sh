#!/usr/bin/env bash

# Shell script to set the ROS_IP and ROS_MASTER_URI evironment
# variables in order to run an experiment using multiple computers.
# Assumes the master will be the tower
# Author: Ian Loefgren


echo $1

if [ $1 == "master" ]; then
	export ROS_IP=192.168.20.111
elif [ $1 == "remote" ]; then
	export ROS_IP=192.168.20.147
	export ROS_MASTER_URI=http://192.168.20.111:11311
else
	:
fi

echo "IPs set"

exit 0
