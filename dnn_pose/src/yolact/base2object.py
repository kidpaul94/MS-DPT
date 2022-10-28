#!/usr/bin/env python  

# ROS & HSR
import rospy
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
import message_filters

# Python Modules
import numpy as np
from scipy.spatial.transform import Rotation
from threading import Lock

# Thead locking
lock = Lock()

def qt2T(data):
    # Convert quaternion and translation to transformation matrix
    quat = np.array([data.pose.pose.orientation.x, data.pose.pose.orientation.y, 
                     data.pose.pose.orientation.z, data.pose.pose.orientation.w])
    T = tf.transformations.quaternion_matrix(quat)
    T[:3,3] = np.array([data.pose.pose.position.x, data.pose.pose.position.y,
                        data.pose.pose.position.z])

    return T


def tracking(pose, pose_prev, init):
    T_pose = qt2T(pose)
    T_pose_inv = np.identity(4)
    T_pose_inv[:3,:3] = np.transpose(T_pose[:3,:3])
    T_pose_inv[:3,3] = -np.matmul(T_pose_inv[:3,:3], T_pose[:3,3])
    
    # Substract current pose from the saved intial pose (pose change)
    delta = np.matmul(pose_prev, T_pose_inv)
    T_final = np.matmul(delta, init)

    return T_final


class odom2obj(object):
    def __init__(self):
        self._init, self._cinit, self._num_prev = np.identity(4), np.identity(4), 0
        self._odom_prev, self._omega_prev, self._track = np.identity(4), np.zeros(3), np.zeros(2)
                                   
        self._odom2obj_pub = rospy.Publisher('/hsrb/object_odom', Odometry, queue_size=10)
        self._IMU2obj_pub = rospy.Publisher('/hsrb/object_imu', Imu, queue_size=10)

        cam_sub = message_filters.Subscriber('/pose_chatter', PoseWithCovarianceStamped)
        num_sub = message_filters.Subscriber('/num_chatter', PointStamped)
        ets = message_filters.TimeSynchronizer([cam_sub, num_sub], 5)
        ets.registerCallback(self._callback1)

        odom_sub = message_filters.Subscriber('/hsrb/odom', Odometry)
        imu_sub = message_filters.Subscriber('/hsrb/base_imu/data', Imu)
        ats = message_filters.ApproximateTimeSynchronizer([odom_sub, imu_sub], 5, 0.2)
        ats.registerCallback(self._callback2)

    def _callback1(self, cam_p, num):
        mid = cam_p.pose.pose
        init_t = [mid.position.x, mid.position.y, mid.position.z]
        init_q = [mid.orientation.x, mid.orientation.y, mid.orientation.z, mid.orientation.w]
        T_obj = tf.transformations.quaternion_matrix(init_q)
        T_obj[:3,3] = init_t
        with lock:
            self._track = [num.point.x, num.point.y]
            self._init = T_obj

    def _callback2(self, odom, imu):
        obj_odom = Odometry()
        obj_IMU = Imu()
        
        with lock:
            if self._track[0] - self._num_prev > 0 and self._track[1] > 0.54:
                # Save the current odom and object pose as a fixed point (a.k.a reinitialization)
                self._odom_prev = qt2T(odom)
                self._cinit = self._init    
            self._num_prev = self._track[0]     

        T_zyx = tracking(odom, self._odom_prev, self._cinit)
        T_zyx = np.ndarray.round(T_zyx, 4)

        # Assign the data as an initial reading for object tracking
        pose = np.zeros(7)
        pose[0:3] = T_zyx[:3,3]
        pose[3:7] = Rotation.from_matrix(T_zyx[:3,:3]).as_quat()
        
        # Transform angular velocity from base frame to object frame
        omega = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z]).reshape((3,1))
        omega_t =  np.matmul(T_zyx[:3,:3], -omega)

        # Transform linear acceleration from base frame to object frame
        '''Currently, IMU transformation does not work well. Temporarily shut it down'''
        imu2base = np.array([[1, 0, 0, -0.003], [0, 1, 0, 0.086],
                             [0, 0, 1, -0.203], [0, 0, 0, 1]])
        imu2obj = np.matmul(T_zyx, imu2base)

        a_linear = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z - 9.80665]).reshape((3,1))
        a_rotation = 25 * (omega - self._omega_prev) # velocity difference / 30Hz (Publishing rate of the IMU sensor)
        self._omega_prev = omega
        a_tangential = np.cross(a_rotation, pose[0:3].reshape((3,1)), axis=0) 
        a_combined = -np.matmul(imu2obj[:3,:3], (a_linear + a_tangential)) 
        print(a_linear)
        print(a_combined)

        # Assign the odom data for publishing
        obj_odom.pose.pose.position.x = pose[0]
        obj_odom.pose.pose.position.y = pose[1]
        obj_odom.pose.pose.position.z = pose[2]
        obj_odom.pose.pose.orientation.x = pose[3]
        obj_odom.pose.pose.orientation.y = pose[4]
        obj_odom.pose.pose.orientation.z = pose[5]
        obj_odom.pose.pose.orientation.w = pose[6]

        # Noise Covariance to fill in.
        obj_odom.pose.covariance = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                    0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 
                                    0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 
                                    0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 
                                    0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 
                                    0.0, 0.0, 0.0, 0.0, 0.0, 0.05]

        # Assign the IMU data for publishing
        obj_IMU.angular_velocity.x = omega_t[0]
        obj_IMU.angular_velocity.y = omega_t[1]  
        obj_IMU.angular_velocity.z = omega_t[2]       
        obj_IMU.linear_acceleration.x = a_combined[0]
        obj_IMU.linear_acceleration.y = a_combined[1]
        obj_IMU.linear_acceleration.z = a_combined[2]

        # Noise Covariance to fill in.
        obj_IMU.angular_velocity_covariance = [0.05, 0.0, 0.0, 
                                               0.0, 0.05, 0.0, 
                                               0.0, 0.0, 0.05]
        obj_IMU.linear_acceleration_covariance = [0.05, 0.0, 0.0, 
                                                  0.0, 0.05, 0.0, 
                                                  0.0, 0.0, 0.05]
        
        # Header for ApproximateTimeSynchronizer
        obj_odom.header.frame_id = odom.header.frame_id
        obj_odom.child_frame_id = odom.child_frame_id
        obj_odom.header.stamp = imu.header.stamp

        obj_IMU.header.frame_id = "base_link"
        obj_IMU.header.stamp = imu.header.stamp

        # Publish
        self._odom2obj_pub.publish(obj_odom)        
        self._IMU2obj_pub.publish(obj_IMU)


if __name__ == '__main__':
    rospy.init_node('hsr_odom2obj') 
    odom2obj()
    rospy.spin()