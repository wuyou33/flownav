Optical Flow Based Navigation
====================
Optical flow based navigation techniques for an indoor surveillance drone.

Assumptions and notes
---------------------
- Pose and trajectory are known (via IMU)
- Scene is mostly static / motion is dominated by camera motion


Notes on Robot Operating System (ROS) 
-------------------------------------
**Installation**

1. Download ROS virtual machine at [RosVM](http://nootrix.com/downloads/#RosVM)

**Basic building**

1. Create project folder (e.g., ~/catkin_ws)
2. Create a folder for dependent packages ~/catkin_ws/src
3. Place source code into ~/catkin_ws/src with e.g. git clone <some_git_repo>
4. Add each package into the package path
> export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/catkin_ws/src/<some_package>
5. Make the packages with catkin
```
> cd ~/catkin_ws
> catkin_make
```
6. Run catkin_make install (not sure this is entirely necessary)
7. Set the path with setup.bash
```
source ~/catkin_ws/devel/setup.bash
```

Now the packages and launch files should be available. For example, for the tum_ardrone package you should now be able to run
```
roslaunch tum_ardrone ardrone_driver.launch
roslaunch tum_ardrone tum_ardrone.launch
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


Previous works
--------------

Camus: Iterative search for "mean" location of flow (center of OF mass)

Match filter: matched focus of expansion filter for angular component
        Pros
        + Doesn't depend on magnitude!
        + weighting by participating components improves robustness
        + a radially increasing weighting may improve even further
        + invariant to rotations

        Issues
        + large search space (but can limit to small search space once found)
        + depends heavily on textured environment