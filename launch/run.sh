#!/usr/bin/env bash

clear
tester="main.py"

#<>TODO: SANITIZE INPUT
#<>TODO: IMPROVE INTERFACE AND MENU, HAVE MENU SHOW UP CONSISTENTLY

# Startup shell script for Core Tools testbed, including ROS stack
echo '--------------------------------------------------------------------'
echo '--------------------------------------------------------------------'
echo '--------------------- Core Tools Testbed ---------------------------'
echo '--------------------------------------------------------------------'
echo '--------------------------------------------------------------------'
echo ' '
echo '*** Experiment parameters can be changed in config.yaml ***'
echo ' '

echo "Using robots:"
if [ $1 == "1" ]; then
    echo "   Deckard"
fi
if [ $2 == "1" ]; then
    echo "   Roy"
fi
if [ $3 == "1" ]; then
    echo "   Pris"
fi
if [ $4 == "1" ]; then
    echo "   Zhora"
fi

use=($1 $2 $3 $4)
robots=("deckard" "roy" "pris" "zhora")
#1 indicates use, positons in arrays match, e.g. use[0]==1 means using deckard
# needs to match config.yaml!


# start roscore and vicon_sys
# xterm -e bash -c "roscore" &
# sleep 2
# xterm -e bash -c "roslaunch ~/vicon_sys.launch" &

# echo "--------------------"
# echo " "
# echo "Log into all robots and run the command:"
# echo "roslaunch cops_and_robots/launch/vicon_nav.launch"
# echo " "

# count=0
# for i in ${robots[@]}; do
#   connection_input=1
#   if [ ${use[$count]} -eq 1 ]; then
#     while [ "$connection_input" != "0" ]; do
#         if [ "$connection_input" != "1" ] ; then
#             xterm -e "ssh odroid@$i" &
#         fi
#         echo " "
#         echo "--------------------"
#         echo "Enter '0' if connection was successful"
#         echo "Enter '1' to retry $i connection"
#         read connection_input
#     done
#   fi
#   count=$count+1
# done

#run code for experiment
echo "When vicon_nav.launch has been started on all robots, press ENTER to run experiment"
read x
run_input=1
while [ "$run_input" != "0" ]
do
    if [ "$run_input" != "1" ]; then
      xterm -hold -e "python $tester"
      echo "-------"
      echo "Enter '1' to re-run experiment"
      echo "or enter '0' to end the program"
    fi
    read run_input
done
echo "Exiting the program..."
exit 0
