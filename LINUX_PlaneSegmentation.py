# Washington State Univeristy: ME 416, Capstone Course
# Project: Los Alamos National Lab: Shape Detection and Path Planning Algorithm
# Team: Shape and Path Planning (SAPP)
# Professor: Chuck Pezeschki, Emily Allen
# Team Members: Kyle Appel, Shea Cooke, Jake Darrow, Petra Jonson, Ashley Sande
# Contacts: kyle.appel@wsu.edu, shea.cooke@wsu.edu, jake.darrow@wsu.edu, petra.jonson@wsu.edu, ashley.sande@wsu.edu 

# NOTE: This can only run on Linux due to PCL dependencies

from __future__ import print_function
import open3d as o3d
import numpy as np
import pcl
import random

pcd_load = o3d.io.read_point_cloud("Example1- Cloud2.ply") #Load in a point cloud file
xyz_load = np.asarray(pcd_load.points) #Create an array with point cloud points

def main():
    cloud = pcl.PointCloud()

    xyzload = np.float32(xyz_load)
    j = 0

    while (len(xyzload)>5): #Loop through creating PLY files of individual sides until there are no more points left
        cloud.from_array(xyzload) #Creating a point cloud from the array

        seg = cloud.make_segmenter() #Segment a single plane from the point cloud
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_distance_threshold(0.1) #Set the threshold for how close points need to be for a plane to be formed
        indices, coefficients = seg.segment()

        if len(indices) == 0:
            print('Could not estimate a planar model for the given dataset.')
            exit(0)

        print('Model coefficients: ' + str(coefficients[0]) + ' ' + str(
            coefficients[1]) + ' ' + str(coefficients[2]) + ' ' + str(coefficients[3]))

        print('Model inliers: ' + str(len(indices)))
        for i in range(0, len(indices)):
            print(str(indices[i]) + ', x: ' + str(cloud[indices[i]][0]) + ', y : ' +
                    str(cloud[indices[i]][1]) + ', z : ' + str(cloud[indices[i]][2]))

        Cloud2 = [] #Create a cloud of just the plane points
        for i in range(0, len(indices)):
            test2 = [cloud[indices[i]][0],cloud[indices[i]][1], cloud[indices[i]][2]]
            Cloud2.append(test2)

        #Export the plane points as a ply file
        pcd = o3d.geometry.PointCloud() 
        pcd.points = o3d.utility.Vector3dVector(Cloud2)
        o3d.io.write_point_cloud(f'test{j}.ply', pcd)
        np.savetxt(f'data{j}.csv', Cloud2, delimiter=',')

        for i in range(0, len(indices)-1): #Remove found planar points from the file
            xyzload = np.delete(xyzload, indices[i]-i, axis = 0)

        j += 1


if __name__ == "__main__":
    main()



