# WSU-SAPP-Point_Cloud_Processing
This code showcases the work done throughout the WSU ME 416: Senior Capstone Project (sponsored by Los Alamos National Lab) to accept and process a point cloud of simple shapes. Initial implementation is designed for creating points offset from a surface such that they could be offloaded to a path planning algorithm for a robotic arm with a gamma detector.

Note: The code beginning with "LINUX" is only operable on Linux based software due to one of the libraries it utilizes only being available within Linux (pcl).

## Use and Operation

This code operates in a two part execution. First, the .PLY file must be processed to separate the object into the respective planes using LINUX_PlaneSegmentation.py. Second, the created .csv must be processed using the SAPP_Plane_Offset.py file.

### Step 1: Processing the Initial .PLY

The .PLY will be separated into different planes based on the specified threshold on line 32. For objects with greater curvature or more faces, this threshold should be reduced.

### Step 2: Processing the .csv

The initial parameters (lines 21-27) must be set prior to running the code. These will specify the necessary paramters that will simplify the point cloud based on the detection_dia, how far the offset desired is, and the check for successfuly scanned points. This will then iterate through the provided file names, detecting their normal equations, and offset them based on the provided parameters.
