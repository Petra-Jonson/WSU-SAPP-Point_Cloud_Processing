# Washington State Univeristy: ME 416, Capstone Course
# Project: Los Alamos National Lab: Shape Detection and Path Planning Algorithm
# Team: Shape and Path Planning (SAPP)
# Professor: Chuck Pezeschki, Emily Allen
# Team Members: Kyle Appel, Shea Cooke, Jake Darrow, Petra Jonson, Ashley Sande
# Contacts: kyle.appel@wsu.edu, shea.cooke@wsu.edu, jake.darrow@wsu.edu, petra.jonson@wsu.edu, ashley.sande@wsu.edu 

import pyransac3d as pyr
import numpy as np
import open3d as o3d
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math as m
import itertools as it
import random as r
import csv
from matplotlib.ticker import FormatStrFormatter

detection_dia = 0.0508
offset_amount = 0.0508
valid_scan_dist = 0.0508
scanning_check = False
start_loc = '' # File location, example: c:/Users/username/Desktop/SAPP_ME416/
loc_append = '' # File name format, example: GarbageCan, this will be appended with a number to automatically generate all file names
amount_of_files = 4 # Number of files, starting at 0. Final file names should be: 'c:/Users/username/Desktop/SAPP_ME416/GarbageCan#'


file_names = []
for i in range(amount_of_files):
    temp_loc = start_loc
    temp_loc = temp_loc + loc_append + str(i) + '.csv'
    file_names.append(temp_loc)

def parse_plane(pc, min_val, max_val, dimension, final_segment = False):
    
    # use whatever dimension was passed in to find the dimension of the index
    segmented_pts = []
    dim_index = 3
    if dimension == 'x': dim_index = 0
    elif dimension == 'y': dim_index = 1
    elif dimension == 'z': dim_index = 2

    # Final segment will be whatever is leftover after dividing the plane into squares of side length "l"
    if final_segment:
        for i in pc:
            if i[dim_index] > min_val:
                segmented_pts.append(i)
        return segmented_pts
    # All non-edge points in the point cloud
    for i in pc:
        if i[dim_index] < max_val and i[dim_index] > min_val:
            segmented_pts.append(i)
    return segmented_pts

def determine_norm_cart_direction(x_range, y_range, z_range, side_l, min_x, min_y, min_z):

    # Define indices for readability
    x, y, z = 0, 1, 2

    # Determine the normal direction based on which cartesian direction has the least variance
    # Find useful information based on the normal direction
    if x_range < y_range and x_range < z_range:
        cartesian_normal = 'x'
        cartesian_normal_index = x
        segments_a = (y_range)/side_l
        segments_b = (z_range)/side_l
        # Can only have a whole number of segments
        segments_a = m.trunc(segments_a)
        segments_b = m.trunc(segments_b)
        # Set the min and max values to be 1/10000 times the overall length of the object to guarantee that ALL points are included
        min_a = min_y - (y_range)/10000
        min_b = min_z - (z_range)/10000
        a = 'y'
        b = 'z'
    elif y_range < x_range and y_range < z_range:
        cartesian_normal = 'y'        
        cartesian_normal_index = y
        segments_a = (x_range)/side_l
        segments_b = (z_range)/side_l
        # Can only have a whole number of segments
        segments_a = m.trunc(segments_a)
        segments_b = m.trunc(segments_b)
        # Set the min and max values to be 1/10000 times the overall length of the object to guarantee that ALL points are included
        min_a = min_x - (x_range)/10000
        min_b = min_z - (z_range)/10000
        a = 'x'
        b = 'z'
    else:
        cartesian_normal = 'z'
        cartesian_normal_index = z
        segments_a = (x_range)/side_l
        segments_b = (y_range)/side_l
        # Can only have a whole number of segments
        segments_a = m.trunc(segments_a)
        segments_b = m.trunc(segments_b)
        # Set the min and max values to be 1/10000 times the overall length of the object to guarantee that ALL points are included
        min_a = min_x - (x_range)/10000
        min_b = min_y - (y_range)/10000
        a = 'x'
        b = 'y'
    
    return segments_a, segments_b, min_a, min_b, a, b, cartesian_normal, cartesian_normal_index #cartesian normal will be used when processing pre-made planes from PLY

def divide_plane(pc, side_l):#TODO: add an average z value for each plane, but need to make sure the avg does not include values from corners.
    # Divide a plane into Local Point Clouds (LPCs) based on a square of side lengths side_l
    LPCs = []

    # Separate point cloud (pc) into xyz components
    x, y, z = zip(*pc)

    # Find the indices of the box that bounds the pc
    min_x = min(x)
    max_x = max(x)
    min_y = min(y)
    max_y = max(y)
    min_z = min(z)
    max_z = max(z)

    # Find the size of the box bounding the pc, aka the range of the points
    x_range = max_x - min_x
    y_range = max_y - min_y
    z_range = max_z - min_z

    # Divide the plane into segments along a and b, where a and b represent cartesian directions (xy, yz, xz)
    segments_a, segments_b, min_a, min_b, a, b, cartesian_normal, cartesian_normal_index = determine_norm_cart_direction(x_range, y_range, z_range, side_l, min_x, min_y, min_z)
    centers = [None]*((segments_a + 1)*(segments_b + 1))
    
    # find the average pt position in thenormal direction, for use with the center of each subdivided plane
    if cartesian_normal_index == 0:
        norm_pt_avg = np.mean(x)
    elif cartesian_normal_index == 1:
        norm_pt_avg = np.mean(x)
    elif cartesian_normal_index == 2:
        norm_pt_avg = np.mean(x)

    # Loop through and divide the plane into a rectangles, then each rectangle into b sub-squares.
    # Store the centers of each square for finding the point that best describes the square
    center_index = 0
    for i in range(segments_a + 1):
        lower_bound_a = min_a + i*side_l
        if i > segments_a:
            a_dim_LPCs = parse_plane(pc, lower_bound_a, lower_bound_a + side_l, a, True)
        else:
            a_dim_LPCs = parse_plane(pc, lower_bound_a, lower_bound_a + side_l, a)
        
        # Create the subdivided point cloud, or Local Point Cloud (LPC)
        # Point of improvement: using %2, alternate between starting at the upper or lower bound to make
        # a zig-zag pattern for the robot
        for j in range(segments_b + 1):
            lower_bound_b = min_b + j*side_l

            # Special condition for the final 
            # if i > segments_b:
            #     LPCs.append(parse_plane(a_dim_LPCs, lower_bound_b, lower_bound_b + side_l, b, True))
            # else:
            LPCs.append(parse_plane(a_dim_LPCs, lower_bound_b, lower_bound_b + side_l, b))
            ab_center = [lower_bound_a + side_l/2, lower_bound_b + side_l/2]
            ab_center.insert(cartesian_normal_index, norm_pt_avg)
            centers[center_index] = ab_center.copy() # might have some merit to using offset_pt
            center_index += 1
            
    return LPCs, centers, cartesian_normal, cartesian_normal_index

def find_closest_pt(lpc, center):# works in xy dimensions, as the center is currently only in xy. The center will eventually use an avg z value
    min_index = 0
    # declare index of x/y/z as numbers for readability
    x, y, z = 0, 1, 2

    # Find the minimum distance between each point in the local point cloud (lpc) and the center of the square that defines it
    min_dist = m.sqrt((lpc[min_index][x] - center[x])**2 + (lpc[min_index][y] - center[y])**2 + (lpc[min_index][z] - center[z])**2)
    for i, pt in enumerate(lpc):
        dist = m.sqrt((pt[x] - center[x])**2 + (pt[y] - center[y])**2 + (pt[z] - center[z])**2)
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return lpc[min_index]

def find_vector_angles(a,b,c,x,y,z):
    # Find the angle between vectors: [x,y,z], and [a,b,c] using the dot-product formula
    cos = (a*x+b*y+c*z)/(m.sqrt(a**2 + b**2 + c**2)*m.sqrt(x**2 + y**2 + z**2))
    theta = m.acos(cos)
    # Theta over 2 for the quaternion
    sin_2 = m.sin(theta/2)
    cos_2 = m.cos(theta/2)
    return cos_2, sin_2

def find_rot_quaternion(a,b,c,x,y,z, sin, cos):
    # Vectors of the form:
    # [a,b,c]
    # [x,y,z]

    # Find the normal vector to both of these through cross-product
    norm_vec = [b*z - c*y, -1*(a*z - c*x), a*y - b*x]

    #Construct a quaternion of form [w,x,y,z],where cos and sin are of theta/2
    quat = [cos, sin*norm_vec[0], sin*norm_vec[1], sin*norm_vec[2]]
    return quat

def pixelize_plane(rotated_plane):
    # Determine the dimension of the squares that fits within the gamma ditector's circumference
    sq_dim = detection_dia/(2**0.5)

    # Find all Local Point Clouds(LPCs) that are part of each square the plane is subdivided into, the centers of these squares, and the index of the normal direction
    LPCs, centers, _, cartesian_normal_index = divide_plane(rotated_plane, sq_dim)

    # Create a list of all points closest to the center within each square
    waypoints = []
    for i, pc in enumerate(LPCs):
        if pc:
            waypoints.append(find_closest_pt(pc, centers[i]).copy())
    
    # Offset all points in the normal direction by the offset amount
    for i in waypoints:
        i[cartesian_normal_index] += offset_amount

    return waypoints, centers

def rot_plane(q, pts):
    rotated_plane = [None]*len(pts)

    # Setup indices as variables for readability
    w,x,y,z = 0,1,2,3

    # Rotate list of pts by quaternion q
    for index, i in enumerate(pts):
        new = [0, i[w], i[x], i[y]]
        rotated_pt_x = (new[x]*(q[w]**2+q[x]**2-q[y]**2-q[z]**2)) + (new[y]*(2*q[x]*q[y] - 2*q[w]*q[z])) + (new[z]*(2*q[x]*q[z]+2*q[w]*q[y]))
        rotated_pt_y = (new[x]*(2*q[x]*q[y]+2*q[w]*q[z])) + (new[y]*(q[w]**2-q[x]**2+q[y]**2-q[z]**2)) + (new[z]*(2*q[y]*q[z]-2*q[w]*q[x]))
        rotated_pt_z = (new[x]*(2*q[x]*q[z]-2*q[w]*q[y])) + (new[y]*(2*q[y]*q[z]+2*q[w]*q[x])) + (new[z]*(q[w]**2-q[x]**2-q[y]**2+q[z]**2))
        rotated_plane[index] = [rotated_pt_x.copy(), rotated_pt_y.copy(), rotated_pt_z.copy()]
    return rotated_plane

def undo_rot(q, pts):

    plane = [None]*len(pts)

    # Setup indices as variables for readability
    w,x,y,z = 0,1,2,3

    #Undo all rotations of each point in list pts by quaternion q
    for index, i in enumerate(pts):
        new = [0, i[w], i[x], i[y]]
        x_pts = (new[x]*(q[w]**2+q[x]**2-q[y]**2-q[z]**2)) + (new[y]*(2*q[x]*q[y] + 2*q[w]*q[z])) + (new[z]*(2*q[x]*q[z] - 2*q[w]*q[y]))
        y_pts = (new[x]*(2*q[x]*q[y] - 2*q[w]*q[z])) + (new[y]*(q[w]**2-q[x]**2+q[y]**2-q[z]**2)) + (new[z]*(2*q[y]*q[z] + 2*q[w]*q[x]))
        z_pts = (new[x]*(2*q[x]*q[z] + 2*q[w]*q[y])) + (new[y]*(2*q[y]*q[z] - 2*q[w]*q[x])) + (new[z]*(q[w]**2-q[x]**2-q[y]**2+q[z]**2))
        plane[index] = [x_pts.copy(), y_pts.copy(), z_pts.copy()]
    
    return plane

def magnitude(vector):
    #root of the sum of the squares of a vector of n size
    temp = m.sqrt(sum(pow(n, 2) for n in vector))
    return temp

def create_offset_points(xyz_pts, obj_avg, plane_avg):

    #Setup plane for pyransac 
    plane1 = pyr.Plane()

    #Find plane equation
    eq, _ = plane1.fit(np.asarray(xyz_pts), 0.005, maxIteration=50000)

    #Find coefficients of plane normal
    a,b,c,d = eq[0],eq[1],eq[2],eq[3] #equation from the first plane

    #Find the rotation nessary to rotate the plane to be normal to xy plane
    x,y,z = 0,0,1 #xy normal vector

    #Find the angles between the vector for use with the quaternion
    cosine_theta_over_2, sine_theta_over_2 = find_vector_angles(a,b,c,x,y,z)

    #Determine the quatrion for rotation to xy normal
    q = find_rot_quaternion(a,b,c,x,y,z, sine_theta_over_2, cosine_theta_over_2)

    #Find the rotated plane
    rotated_plane = rot_plane(q, xyz_pts)#[None]*len(xyz_pts)

    #Find points by subdividing the plane into squares
    offset_in_rotated_frame, centers = pixelize_plane(rotated_plane)

    #Undo the rotation to get the offset from the object in the original reference frame
    offset_pts_in_original_ref_frame = undo_rot(q, offset_in_rotated_frame)


    if determine_to_reprocess(obj_avg, plane_avg, offset_pts_in_original_ref_frame):
        offset_pts_in_original_ref_frame = reprocess_plane(offset_in_rotated_frame, q)
    
    # Return all points for plotting
    return rotated_plane, offset_in_rotated_frame, offset_pts_in_original_ref_frame#, plane_returned_to_normal <- for verifying quaternion rotation

def determine_to_reprocess(obj_avg, plane_avg, xyz_pts):
    
    xl, yl, zl = zip(*xyz_pts)
    avg_waypoint = [np.mean(xl), np.mean(yl), np.mean(zl)]

    # Determine the planes to reprocess based on if the average offset point is closer to the center than te matching plane of the object
    diff_rc = [avg_waypoint[0] - obj_avg[0], avg_waypoint[1] - obj_avg[1], avg_waypoint[2] - obj_avg[2]]
    diff_pc = [plane_avg[0] - obj_avg[0], plane_avg[1] - obj_avg[1], plane_avg[2] - obj_avg[2]]
    if magnitude(diff_rc) < magnitude(diff_pc):
        return True
    else:
        return False

def reprocess_plane(plane, q):
    # accepts the rotated plane and the associated quaternion
    # Reprocess the necessary planes by offset twice the ammount in the opposite direction
    rotated_corrected_plane = [None]*len(plane)
    for j, pt in enumerate(plane):
        rotated_corrected_plane[j] = [pt[0], pt[1], pt[2] - 2*offset_amount]

    corrected_plane = undo_rot(q, rotated_corrected_plane)

    return corrected_plane

def read_csv(file_name):
    x_csv = []
    y_csv = []
    z_csv = []

    #Read in all points from the csv file. Format: "<float>,<float>,<float>""
    with open(file_name, 'rt') as f:
        reader = csv.reader(f, delimiter = ',')
        for row in reader:
            if row:
                #Read in each value of the row as a string and immediately type-cast it as a float
                x_csv.append(float(row[0]))
                y_csv.append(float(row[1]))
                z_csv.append(float(row[2]))

    # Separate csv points into each x/y/z component
    num_pts = len(x_csv)
    
    xyz_pts = [None]*num_pts
    for i in range(num_pts):
        xyz_pts[i] = [x_csv[i], y_csv[i], z_csv[i]]

    return xyz_pts

def ScanCheck(obj_pts, robo_pts):
    Scanned = []
    for n, i in enumerate(robo_pts):
        for j in obj_pts:
            diff = [0,0,0]
            diff[0] = i[0] - j[0]
            diff[1] = i[1] - j[1]
            diff[2] = i[2] - j[2]

            if magnitude(diff) < valid_scan_dist: #Value of Scan Radius
                Scanned.append(j.copy())
    return Scanned

def offset_and_plot(all_file_names):
    rotated_plane = []
    waypoints_separated = []
    xyz_pts = []

    all_pts = []
    plane_end_indices = []
    plane_start_indices = []
    plane_avgs = []
    #Loop through to make all point lists for plotting everything as an example 
    for i in all_file_names:
        xyz_pts = read_csv(i)
        plane_start_indices.append(len(all_pts))
        all_pts.extend(xyz_pts)
        plane_end_indices.append(len(all_pts) - 1)
        plane_x, plane_y, plane_z = zip(*xyz_pts)
        plane_avgs.append([np.mean(plane_x), np.mean(plane_y), np.mean(plane_z)])
    
    obj_x, obj_y, obj_z = zip(*all_pts)
    obj_avg = [np.mean(obj_x), np.mean(obj_y), np.mean(obj_z)]

    all_offset = []
    for j in range(len(plane_start_indices)):
        xy_norm_planes, offset_rotated, offset_pts = create_offset_points(all_pts[plane_start_indices[j]:plane_end_indices[j]], obj_avg, plane_avgs[j])
        rotated_plane.extend(xy_norm_planes.copy())
        waypoints_separated.extend(offset_rotated.copy())
        all_offset.extend(offset_pts)

    # all_offset are the points to be passed to the robot
    # offset_pts are serparated by plane
    
    # Everything below, within offset_and_plot is for demonstration purposes

    x_pts, y_pts, z_pts = zip(*all_pts)
    offset_x, offset_y, offset_z = zip(*all_offset)
    xr, yr, zr = zip(*rotated_plane)
    offset_r_x, offset_r_y, offset_r_z = zip(*waypoints_separated)
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title("Offset Object")
    # ax.scatter(x_pts, y_pts, z_pts, color = 'r')
    ax.scatter(offset_x, offset_y, offset_z, color = 'b')
    ax.scatter(obj_avg[0], obj_avg[1], obj_avg[2], color = 'y')
    plt.show(block = False)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(xr, yr, zr, color = 'r')
    ax.scatter(offset_r_x, offset_r_y, offset_r_z, color = 'b')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Rotated to XY Normal")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show(block = False)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    ax.scatter(x_pts, y_pts, z_pts, color = 'r')
    # ax.scatter(offset_x, offset_y, offset_z, color = 'b')
    ax.set_title("Garbage Can - Isolated Plane")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show(block = True)

    if scanning:
        Scanned  = ScanCheck(all_pts, all_offset)
        Scanned_x, Scanned_y, Scanned_z = zip(*Scanned)
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        ax.scatter3D(obj_x, obj_y, obj_z, s = 10, color = 'blue', marker = 'o')
        ax.scatter3D(Scanned_x, Scanned_y, Scanned_z, s = 40, color = 'green', marker = 's')
        ax.set_title('Pointcloud, Scanned Points, and Scanning Positions')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend(['Point Cloud', 'Successfully Scanned Points'])
        plt.show(block = True)


offset_and_plot(file_names)
print("beakpoint")