![header](https://capsule-render.vercel.app/api?type=rect&color=timeGradient&text=NAVLER%20LABS%20CHALLENGE%20"2020"&fontSize=20)

## <div align=left>:heavy_plus_sign:REPO INFO</div>  
- KAIST IRiS Lab. Autonomous Vehicle "PHAROS" 
- VEHICLE SENSOR PACKAGE  

## <div align=left>:heavy_plus_sign:REPO CONFIG</div>  
#### PHAROS/MSG  
* ROS based VEHICLE COM MSG   
#### SENSOR IMU  
* ebimu driver      : EB-IMU V5 9-DoF Low-Cost IMU SENSOR  
* xsense ros driver : XSENSE MTI-30 9-DOF AHRS IMU SENSOR  
#### SENSOR LiDAR  
* MATLAB : MATLAB SUB SCRIPT
* RGB LiDAR : FUSING LiDAR + RGB VISION  
* VELODYNE LiDAR : Velodyne VLP 16 Package with 3 LiDAR  
#### SENSOR TF  
* ROS TF : Transformation between Sensors  
#### SENSOR VISION
* blckfly S FLIR : FLIR BlackFly S USB STEREO VISION  
* Camera_umd     : UVC CAMERA for Logitech BRIO 4K 


## <div align=left>:heavy_plus_sign:REPO USE</div> 
<pre>cd catkin_ws/  
git clone https://github.com/iismn/AGV-SENSOR-PACK  
cd .. && catkin_make</pre>

## <div align=left>:heavy_plus_sign:ADD INFO</div>
#### VEHICLE CONSIST SENSOR 
- Velodyne VLP-16 Hi-Res LiDAR x3  
- Xsense MTI-30 AHRS IMU x1  
- EB-IMU V5 AHRS IMU x1  
- FLIR BlackFly S USB VISION x2  
- Logitech BRIO 4K USB VISION x1  
- Ublox M8P RTK-GPS x1
- <del>Novatel RTK GPS+INS x1</del>
- <del>Velodyne HDL-32 Hi-Res LiDAR x1</del>
