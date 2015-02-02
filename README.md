# Obstacle Avoidance Using Expansion Estimation

This project implements an interpretation of the algorithm developed by Mori and Scherer (paper [here](https://www-preview.ri.cmu.edu/pub_files/2013/5/monocularObstacleAvoidance.pdf)) for frontal obstacle detection using expansion cues from a monocular camera. The project code is integrated into the Robot Operating System architecture (ROS), allowing for a modular design that combines several components into a distributed framework.

## Assumptions for Obstacle Avoidance

+ The drone moves along the Z-axis at a constant velocity with yaw/pitch velocities being negligible.

+ Obstacles are small relative to image area i.e., fit well within the imaged scene.

+ Obstacles are static or motion is dominated by camera motion.

+ Obstacles are sufficiently textured to be reliably detected by SURF algorithm.

## Derivation

It is straightforward to show that the rate of change of scale of an object
measured between two image frames is inversely related the distance to the
object. Assuming that the drone is headed straight towards a point on an
obstacle, we can estimate the Time to Contact (TTC) with an object using only
the rate of change in scale of an obstacle and the time interval between two
frames. Figure 1 below shows the derivation of this relationship assuming the
the pinhole camera model.

![Estimating TTC](ttc_diagram.png "Estimating TTC")

In other words,

![TTC](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7Bf%7D%7Bcw%7D%20%3D%20%5Cfrac%7BBC%7D%7BW%7D%20%2C%20%5Cfrac%7Bf%7D%7Bw%7D%20%3D%20%5Cfrac%7BAC%7D%7BW%7D%20%5CLongrightarrow%20t_%7BBC%7D%3Dt_%7BAB%7D%20.%20%5Cfrac%7B1%7D%7Bc-1%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

An alternate formulation gives the distance to object using the viewing angle and estimated distance between A and B

![Distance](http://www.sciweavers.org/tex2img.php?eq=BC%3D%5Cfrac%7BAB%5Ctan%7B%5Calpha%7D%7D%7B%5Ctan%7B%5Cbeta%7D-%5Ctan%7B%5Calpha%7D%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)


## Related works

### Camus, '94
Real Time Optical Flow
ftp://ftp.cs.brown.edu/pub/techreports/94/cs94-36.pdf

Camus developed one of the first successful applications where optical vision was used for obstacle avoidance in real time. He developed a block matching optical flow algorithm that estimated flow using reduced computational resources by finding the best match across _several_ time adjacent frames within a small spatial neighborhood. Camus solved the problem of obstacle detection by estimating the _time_ _to_ _contact_ (TTC) derived using an estimate of the location of the Focus of Expansion (FoE): the point around which optical flow radially diverges. Assuming that the camera is headed directly towards the FoE, it can be shown that the time to collision with an object is inversely related to the divergence of the optical flow around the FoE. By assuming translational motion and only one obstacle in the scene, the FoE location was estimated by simply averaging optical flow over the entire image without respect to flow magnitude and iteratively reducing the window size to refine the measurement until it was reduced to a 4x4 window. Camus performed experiments with a wheeled mobile robot which showed good convergence to true TTC. However, the environment was limited to an indoor scene where the robot maintained a constant velocity of 5cm/s and where unwanted movements of the robot (such as vibrations and momentary rotations) were truly negligible.

### Zufferey, '05
Toward 30-gram autonomous indoor aircraft: Vision based obstacle avoidance and altitude control

Zufferey developed a biologically inspired light weight airplane flyer with a few low res 1-D cameras and onboard microcontroller that could explore a highly textured indoor environment without colliding into walls. The mode of information for obstacle avoidance here was decidedly simpler. Only the divergence of optical flow was estimated in (1-D) regions of the image. The flyer used a balancing strategy to maintain its pitch and yaw by differencing the optical flow on the left and right of a horizontally oriented line sensor and differencing the top and bottom half of flow with a vertically oriented line sensor. The horizontal difference was also subject to a threshold which, by estimating flow divergence, could trigger a 90 degree saccade to avoid frontal obstacles. Noting that inevitable rotational optical flow caused by the flyer's steering introduced spurious measurements that are unrelated to the distance to obstacle, Zufferey also used an IMU to monitor and remove effects of rotations.

### Sazbon, '04
Finding the focus of expansion and estimating range using optical flow images and a matched filter

Sazbon proposes a unique yet simple approach to focus of expansion detection, a subproblem of the time to contact method of obstacle avoidance. A filter is proposed that is simply a template of the focus of expansion - a square window with perfectly radially diverging optical flow with zero flow at the center. In matching, only the angle of optical flow is considered so that matching is invariant to flow magnitude. However, the experiments only include focus of expansion location on a few ideal images and algorithm wasn't integrated into any actual application e.g. obstacle avoidance for a mobile robot.

## Implementation details

Feature tracking
clustering of features
removal of poor matches
improving estimation using multiple scale estimates
factors for selecting/rejecting objects

## Dependencies

- `ardrone_autonomy`

- `opencv`


## Optional

- `uvc_camera`

- `ros-joy`

## Notes on Robot Operating System (ROS) 

**Installation**

1. Download ROS virtual machine at [RosVM](http://nootrix.com/downloads/#RosVM) or install the ros-packages onto your system (see the ROS installation instructions on their wiki).

**Step for building a catkin project**

1. Create project folder (e.g., ~/catkin_ws)

2. Create a folder for dependent packages ~/catkin_ws/src

3. Place source code into ~/catkin_ws/src with e.g. git clone <some_git_repo>

4. Make the packages with catkin

    ```bash
    $ cd ~/catkin_ws
    $ catkin_make
    ```

7. Set the path with setup.bash
    ```bash
    $ . ~/catkin_ws/devel/setup.bash
    ```

Now the packages and launch files should be available. For example, for the flownav package you could run
```bash
$ roscore >& roscore.log &          # make sure roscore is started
$ roslaunch flownav flownav.launch
```

Good deal!


## Using screen

Emulates multiple terminals using one terminal session. Really useful when you
are running ROS on a VM with several processes outputting to the terminal that
you would like to monitor. Good intro here:
http://nathan.chantrell.net/linux/an-introduction-to-screen/

* C-a a to go to start of line
* C-a C-a to toggle between previously used window or C-n/C-p to tab forward/backward
* C-a c to create a new window
* C-a w to list active windows
* screen <some program> to run a program without opening an intermediate shell
* C-a k to kill a window
* C-a d to detach session
* screen -r to connect to open session


## Setting up virtualbox ssh server

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
