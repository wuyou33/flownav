Obstacle Avoidance Using Expansion Estimation
=============================================

This project implements an algorithm by Mori and Scherer (paper [here](https://www-preview.ri.cmu.edu/pub_files/2013/5/monocularObstacleAvoidance.pdf)) for obstacle detection using expansion cueues from a monocular camera. The project code is integrated into the Robot Operating System architecture (ROS), allowing for a modular design that combines several components in an elegant fashion. Modules include

1. uvc_camera       - for streaming video feed from webcam for testing purposes
2. ardrone_autonomy - for communication with and control of the AR Drone
3. opencv           - for feature detection and matching

Assumptions
-----------

If we assume a rectangular coordinate system we can represent a scene by having the origin at the drone's location and moving with constant velocity alone the Z axis.

Pose and trajectory are known via IMU. In particular, velocity and position are known.

Obstacles are static / motion is dominated by camera motion.


Additional notes and instructions (mostly for my own reference) are cited below covering thing's I've learned while doing this project.


Previous works
--------------

Camus: Iterative search for mean location of flow ("center of mass" of optical flow)

Match filter: matched focus of expansion filter for angular component
        Pros
        + Doesn't depend on magnitude!
        + invariant to rotations
        + weighting by participating components improves robustness
        + a radially increasing weighting may improve even further

        Issues
        + large search space (but can limit to small search space once found)
        + depends heavily on textured environment


Notes on Robot Operating System (ROS) 
-------------------------------------
**Installation**

1. Download ROS virtual machine at [RosVM](http://nootrix.com/downloads/#RosVM) or install the ros-packages onto your system (see the ROS installation instructions on their wiki).

**Step for building a catkin project**

1. Create project folder (e.g., ~/catkin_ws)
2. Create a folder for dependent packages ~/catkin_ws/src
3. Place source code into ~/catkin_ws/src with e.g. git clone <some_git_repo>
4. Add each package into the package path
> export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws/src/<some_package>
5. Make the packages with catkin
```
$ cd ~/catkin_ws
$ catkin_make
```
6. Run catkin_make install (not sure this is entirely necessary)
7. Set the path with setup.bash
```
$ . ~/catkin_ws/devel/setup.bash
```

Now the packages and launch files should be available. For example, for the flownav package you could run
```
roslaunch flownav joystick.launch
roslaunch flownav flownav.launch
```
Good deal!


Using screen
------------
Good intro here: http://nathan.chantrell.net/linux/an-introduction-to-screen/

  * C-a a to go to start of line
  * C-a C-a to toggle between previously used window or C-n/C-p to tab forward/backward
  * C-a c to create a new window
  * C-a w to list active windows
  * screen <some program> to run a program without opening an intermediate shell
  * C-a k to kill a window
  * C-a d to detach session
  * screen -r to connect to open session


Setting up virtualbox ssh server
--------------------------------

1. sudo apt-get install open-ssh
2. Go to virtualbox main preferences
  1. Click network
  2. Click Host-only networks tab
  3. Add an adapter
3. Go to machine settings
  1. Click network
  2. Click Adapter 2 tab
  3. Set "Attached to" to Host-Only Adapter
  4. Restart machine
4. Edit /etc/network/interfaces and add something similar to the following
```
auto eth1
iface eth1 inet static
address 192.168.56.2
netmask 255.255.255.0
network 192.168.56.0
broadcast 192.168.56.255
```
5. Open /etc/hosts _inside the host machine_ and enter the line
```
<IP>    <hostname>
```
