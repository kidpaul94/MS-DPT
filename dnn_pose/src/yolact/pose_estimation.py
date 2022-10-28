#!/usr/bin/env python

# ROS & HSR
from cv_bridge.core import CvBridgeError
from cv_bridge import CvBridge
import rospy
import tf
from sensor_msgs.msg import Image as ROS_Image
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
import message_filters

# Depth to PC & OBB_based initialization with ICP (OBB: Oriented Bounding BOx)
import open3d as o3d
from scipy.spatial.transform import Rotation
import copy

# YOLACT
from data import COLORS
from yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils import timer
from utils.functions import SavePath
from layers.output_utils import postprocess, undo_image_transformation
from data.config  import cfg, set_cfg

# Python Modules
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import argparse
import os
from collections import defaultdict
import cv2

# Camera intrinsic values
fx = 535.2639891915636
fy = 536.0244886780657
px = 317.4524529077298
py = 241.5520765202342

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='./weights/yolact_plus_resnet50_10_10000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=3, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--config', default=None,
                        help='The config object to use.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--score_threshold', default=0.90, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--pose_visualize', default=False, type=str2bool,
                        help='Whether or not to display pose estimation result')

    parser.set_defaults(no_bar=False, display=False, benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False)

    global args
    args = parser.parse_args(argv)


color_cache = defaultdict(lambda: {})

# YOLACT++ 
def prep_display(dets_out, img, h, w, undo_transform=True, class_color=False, mask_alpha=0.45, fps_str=''):
    # If undo_transform=False then im_h and im_w are allowed to be None.
    if undo_transform:
        img_numpy = undo_image_transformation(img, w, h)
        img_gpu = torch.Tensor(img_numpy).cuda()
    else:
        img_gpu = img / 255.0
        h, w, _ = img.shape
    
    with timer.env('Postprocess'):
        save = cfg.rescore_bbox
        cfg.rescore_bbox = True
        t = postprocess(dets_out, w, h, visualize_lincomb = args.display_lincomb,
                                        crop_masks        = args.crop,
                                        score_threshold   = args.score_threshold)
        cfg.rescore_bbox = save
        
    with timer.env('Copy'):
        idx = t[1].argsort(0, descending=True)[:args.top_k]
        
        if cfg.eval_mask_branch:
            masks = t[3][idx]
        classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

    num_dets_to_consider = min(args.top_k, classes.shape[0])
    for j in range(num_dets_to_consider):
        if scores[j] < args.score_threshold:
            num_dets_to_consider = j
            break


    def get_color(j, on_gpu=None):
        global color_cache
        color_idx = (classes[j] * 5 if class_color else j * 5) % len(COLORS)
        
        if on_gpu is not None and color_idx in color_cache[on_gpu]:
            return color_cache[on_gpu][color_idx]
        else:
            color = COLORS[color_idx]
            if not undo_transform:
                color = (color[2], color[1], color[0])
            if on_gpu is not None:
                color = torch.Tensor(color).to(on_gpu).float() / 255.
                color_cache[on_gpu][color_idx] = color
            return color

    if args.display_masks and cfg.eval_mask_branch and num_dets_to_consider > 0:
        masks = masks[:num_dets_to_consider, :, :, None]

        colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
        masks_color = masks.repeat(1, 1, 1, 3) * colors * mask_alpha

        inv_alph_masks = masks * (-mask_alpha) + 1

        masks_color_summand = masks_color[0]
        if num_dets_to_consider > 1:
            inv_alph_cumul = inv_alph_masks[:(num_dets_to_consider-1)].cumprod(dim=0)
            masks_color_cumul = masks_color[1:] * inv_alph_cumul
            masks_color_summand += masks_color_cumul.sum(dim=0)

        img_gpu = img_gpu * inv_alph_masks.prod(dim=0) + masks_color_summand

    '''Debugging purpose'''
    # img_numpy = (img_gpu * 255).byte().cpu().numpy()
    
    return num_dets_to_consider, masks 


# YOLACT++ 
def evalimage(net, rgb_image):
    frame = torch.from_numpy(rgb_image).cuda().float()
    batch = FastBaseTransform()(frame.unsqueeze(0))
    preds = net(batch)

    test, masks = prep_display(preds, frame, None, None, undo_transform=False) 

    return test, masks 


# YOLACT++ & ROS node initialization
def evaluate_ros(net):
    net.detect.use_fast_nms = args.fast_nms
    net.detect.use_cross_class_nms = args.cross_class_nms
    cfg.mask_proto_debug = args.mask_proto_debug

    # Generate CAD point cloud
    mesh = o3d.io.read_triangle_mesh('./CAD_ply/MDP_TrainingModel.ply')
    target = mesh.sample_points_uniformly(number_of_points = 2048)
    save_normal = target.normals
    pts_target = np.asarray(target.points)
    zero_mean = pts_target - pts_target.mean(axis=0, keepdims=True)
    max_dist = max(np.linalg.norm(zero_mean, axis=1))
    target_norm = zero_mean / max_dist

    target.points = o3d.utility.Vector3dVector(target_norm)
    target.normals = save_normal

    # Calculate OBB and its vertices
    obb1 = target.get_oriented_bounding_box()
    edge_target = np.asarray(obb1.get_box_points())
    rotationax1, target_vertice = get_vertice(edge_target)

    # Target point cloud from CAD
    rospy.init_node('SubePub', anonymous=True)
    ImageFeed(net, target, obb1, max_dist, rotationax1, target_vertice) 
    rospy.spin()

def axis_visualizer(image, mask, RT, stamp):
    try:
        solutions = np.argwhere(mask[0] != 0)
        mean_x = int(np.mean(solutions[:,1]))
        mean_y = int(np.mean(solutions[:,0]))
        mean = (mean_x, mean_y)

        axis = np.int16([[40,0,0], [0,40,0], [0,0,40]])
        rotated_axis = np.dot(RT[:3,:3], axis)
        projection = rotated_axis[:2,:3]
        projection[0,:3] = projection[0,:3] + mean_x
        projection[1,:3] = projection[1,:3] + mean_y
        projection = projection.astype(int)

        # project 3D points to image plane
        image = cv2.line(image, mean, tuple(projection[:2,0].ravel()), (0,0,255), 2)
        image = cv2.line(image, mean, tuple(projection[:2,1].ravel()), (0,255,0), 2)
        image = cv2.line(image, mean, tuple(projection[:2,2].ravel()), (255,0,0), 2)
    except Exception:
        pass

    cv2.imwrite('/home/bartonlab-user/workspace/src/dnn_pose/src/output_images/recording3/masked_' + str(stamp) + '.png', image)


# ROS subscriber class for rgb and depth image from the HSR. Use message_filter to synchronize 2 subscribers.
class ImageFeed(object):
    def __init__(self, net, target, obb1, max_dist, rotationax1, target_vertice):     
        self._net = net
        self._obb, self._target, self._maxd = obb1, target, max_dist
        self._rotax, self._vertice = rotationax1, target_vertice
        self._pose_pub = rospy.Publisher('/pose_chatter', PoseWithCovarianceStamped, queue_size=10)
        self._num_pub = rospy.Publisher('/num_chatter', PointStamped, queue_size=10)
        self._score, self._prevT, self._prevp, self._prevq = 0, np.identity(4), np.zeros(3), np.zeros(4)

        # Synchronize RGB and depth images subscription
        self._tf_sub = tf.TransformListener()
        rgb_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_rect_color', ROS_Image)
        depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_rect_raw', ROS_Image)
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 5, 0.2)
        ts.registerCallback(self._callback)

    def _callback(self, rgb, depth):
        try:
            (trans, rot) = self._tf_sub.lookupTransform('base_link', 'head_rgbd_sensor_link', rospy.Time(0))
            rgb_data = CvBridge().imgmsg_to_cv2(rgb, "rgb8")
            depth_data = CvBridge().imgmsg_to_cv2(depth, "16UC1")
            num_obj, mask = evalimage(self._net, rgb_data)
            mask_np = mask.detach().cpu().numpy()

            pose = PoseWithCovarianceStamped()
            translation = []
            quat = []
            if num_obj != 0:
                source_pc, Tx, Ty, Tz, obb2 = Depth2PC(mask_np, rgb_data, depth_data, num_obj, self._obb, self._maxd)
                if self._score > 0.54:
                    print("Warm starts!\n")
                    RT_xyz = RotEst_warmstart(source_pc, self._target, self._prevT)
                else:
                    print("Intial rotation estimation starts!\n")
                    RT_xyz = RotEst_init(source_pc, self._target, self._obb, obb2, self._rotax, self._vertice)

                self._score = score(source_pc.transform(RT_xyz), self._target)

                # Convert the representation from xyz(Open3D) to zyx
                RT_zyx = copy.copy(RT_xyz)
                RT_zyx[:3,:3] = np.transpose(RT_zyx[:3,:3])
                RT_zyx[:3,3] = (RT_zyx[:3,3] * self._maxd + [Tx, Ty, Tz]) / 1000

                # Perform axis transformation with given data
                quad_tf = np.array(rot)
                mat_tf = tf.transformations.quaternion_matrix(quad_tf)
                mat_tf[:3,3] = trans
                mat_tf = np.ndarray.round(mat_tf, 4)

                base_obj = np.matmul(mat_tf, RT_zyx) # switching the order maybe?
                base_obj = np.ndarray.round(base_obj, 4)
                translation = [base_obj[0,3], base_obj[1,3], base_obj[2,3]]
                quat = Rotation.from_matrix(base_obj[:3,:3]).as_quat()

                self._prevT = RT_xyz
                self._prevp = [base_obj[0,3], base_obj[1,3], base_obj[2,3]]
                self._prevq = quat
                
                # Assign small covariance
                cov_t = 0.027 / self._score
                cov_r = 0.001240 / self._score**6
                pose.pose.covariance = [cov_t, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                        0.0, cov_t, 0.0, 0.0, 0.0, 0.0, 
                                        0.0, 0.0, cov_t, 0.0, 0.0, 0.0, 
                                        0.0, 0.0, 0.0, cov_r, 0.0, 0.0, 
                                        0.0, 0.0, 0.0, 0.0, cov_r, 0.0, 
                                        0.0, 0.0, 0.0, 0.0, 0.0, cov_r]
                    
            else:
                rospy.loginfo('No detected object(s)!\n')
                translation = self._prevp 
                quat = self._prevq
                self._score = 0

                # Inflate the covariance since we are guessing blindly
                pose.pose.covariance = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                                        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 
                                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 
                                        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
                                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 
                                        0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

            # Save data for next iteration
            change_num = PointStamped()
            change_num.point.x = num_obj
            change_num.point.y = self._score

            # Assign variables for publishing
            pose.pose.pose.position.x = translation[0]
            pose.pose.pose.position.y = translation[1]
            pose.pose.pose.position.z = translation[2]
            pose.pose.pose.orientation.x = quat[0]
            pose.pose.pose.orientation.y = quat[1]
            pose.pose.pose.orientation.z = quat[2]
            pose.pose.pose.orientation.w = quat[3]

            print("Position:")
            print(translation)
            print("Orientation:")
            print(quat)

            pose.header.frame_id = 'base_link'
            pose.header.stamp = rgb.header.stamp
            change_num.header.frame_id = 'base_link'
            change_num.header.stamp = rgb.header.stamp
            self._num_pub.publish(change_num)
            self._pose_pub.publish(pose)

            # if args.pose_visualize:
            #     axis_visualizer(img_numpy, mask_np, RT_zyx, pose.header.stamp)

        except (CvBridgeError, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print(e)


# Depth image to point cloud using the mask from YOLACT++
def Depth2PC(mask, img, depth_map, num_obj, obb1, maxd):
    # Currently Depth2PC only take single object (i.e., mask[0])
    # If needed to take multiple objects, change the code so that it can process mask[i]
    source, Tx, Ty, Tz, obb2 = o3d.geometry.PointCloud(), 0, 0, 0, 0
    
    for i in range(num_obj):
        solutions = np.argwhere(mask[i] != 0)

        # Edge Detection
        gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, np.ones((2,2),np.uint8))
        im_bw = cv2.cvtColor(gradient, cv2.COLOR_RGB2GRAY)

        seg = im_bw[solutions[:,0], solutions[:,1]]
        idx1 = np.argwhere(seg[...] > 40)
        solutions_filtered = np.delete(solutions, idx1[:,0], axis = 0)

        # Depth image from HSR in 16UC1 format (unit of the raw value = mm)
        kernel = np.ones((5,5),np.uint8)
        opening = cv2.morphologyEx(depth_map, cv2.MORPH_OPEN, kernel)

        # Conversion from pixel location of the object in the depth image to camera frame coordinate
        pts = np.zeros((3,solutions_filtered.shape[0]))
        pts[0] = (solutions_filtered[:,1] - px)/fx * opening[solutions_filtered[:,0],solutions_filtered[:,1]]
        pts[1] = (solutions_filtered[:,0] - py)/fy * opening[solutions_filtered[:,0],solutions_filtered[:,1]] 
        pts[2] = opening[solutions_filtered[:,0],solutions_filtered[:,1]]

        # Delete 0 depth
        idx2 = np.argwhere(np.any(pts[...,:] == 0, axis = 0))
        pts_filtered = np.delete(pts, idx2, axis = 1)

        # Object position in camera coordinate
        num_total = pts_filtered.shape[1] 
        Tx = np.sum(pts_filtered[0])/num_total 
        Ty = np.sum(pts_filtered[1])/num_total
        Tz = np.sum(pts_filtered[2])/num_total

        # Write .ply file
        source.points = o3d.utility.Vector3dVector(pts_filtered.T)

        pts_source = np.asarray(source.points)
        zero_mean_source = pts_source - pts_source.mean(axis=0, keepdims=True)
        source_norm = zero_mean_source / maxd
        source.points = o3d.utility.Vector3dVector(source_norm)

        # Downsample with a voxel size and Remove outlier in observed pc 
        source = source.voxel_down_sample(voxel_size=0.06)
        source, _ = source.remove_statistical_outlier(nb_neighbors=200, std_ratio=2.0)

        try:
            obb2 = source.get_oriented_bounding_box()
            source.translate(obb1.get_center() - obb2.get_center())
            obb2.translate(obb1.get_center() - obb2.get_center())
        except Exception:
            pass
        
    return source, Tx, Ty, Tz, obb2
    

# OBB-based initialization with ICP (OBB: Oriented Bounding Box)
def RotEst_init(source, target, obb1, obb2, rotationax1, target_vertice):
    ROT = [0]*6
    ROT[0] = np.eye(3)
    ROT[1] = Rotation.from_rotvec(np.pi / 6 * target_vertice).as_matrix()
    ROT[2] = Rotation.from_rotvec(np.pi / 3 * target_vertice).as_matrix()
    ROT[3] = Rotation.from_rotvec(np.pi / 2 * target_vertice).as_matrix()
    ROT[4] = Rotation.from_rotvec(2 * np.pi / 3 * target_vertice).as_matrix()
    ROT[5] = Rotation.from_rotvec(5 * np.pi / 6 * target_vertice).as_matrix()
    trans_comb = np.eye(4)
    source_1 = copy.deepcopy(source)
    
    try:
        # Actual algorithm
        edge_source = np.asarray(obb2.get_box_points())
        _, source_vertice = get_vertice(edge_source)

        source_1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
        source_1.normalize_normals()
        normal = np.asarray(source_1.normals)
        is_single = np.dot(normal[1:], normal[0])
        is_normal = np.argwhere(np.logical_and(is_single<=0.02, is_single>=-0.02))
        volume_ratio = obb2.volume() / obb1.volume()
        arm_length = 0
        init_trans = np.eye(4)

        if volume_ratio < 0.5:
            print("Insufficient data to process 6D pose! Volue ratio = 0.5")
            mean_normal1 = np.asarray(source_1.normals).mean(0)
            init_trans[:3,:3] = rotation_matrix(mean_normal1, target_vertice)
            init_trans[:3,3] = -0.5 * target_vertice
            source_1.transform(init_trans)
        else:
            init_trans[:3,:3] = rotation_matrix(source_vertice, target_vertice)
            source_1.transform(init_trans)
            source_1.orient_normals_consistent_tangent_plane(35)
            mean_normal1 = np.asarray(source_1.normals).mean(0)
            # if is_normal.size == 0:
            #     print("Insufficient data to process 6D pose!")
            #     arm_length = 0.75

        source_2 = copy.deepcopy(source_1)
        R8 = Rotation.from_rotvec(np.pi * rotationax1).as_matrix()
        init_trans2 = Rt2T(R8, np.array([0, 0, 0]))
        source_2.transform(init_trans2)
        mean_normal1 = arm_length * mean_normal1
        mean_normal2 = arm_length * np.asarray(source_2.normals).mean(0)

        ICP1 = [0]*6
        TRANS1 = [0]*6
        ICP1[0], TRANS1[0] = registration_result(source_1, target, mean_normal1, ROT[0])
        ICP1[1], TRANS1[1] = registration_result(source_1, target, mean_normal1, ROT[1])
        ICP1[2], TRANS1[2] = registration_result(source_1, target, mean_normal1, ROT[2])
        ICP1[3], TRANS1[3] = registration_result(source_1, target, mean_normal1, ROT[3])
        ICP1[4], TRANS1[4] = registration_result(source_1, target, mean_normal1, ROT[4])
        ICP1[5], TRANS1[5] = registration_result(source_1, target, mean_normal1, ROT[5])

        ICP2 = [0]*6
        TRANS2 = [0]*6
        ICP2[0], TRANS2[0] = registration_result(source_2, target, mean_normal2, ROT[0])
        ICP2[1], TRANS2[1] = registration_result(source_2, target, mean_normal2, ROT[1])
        ICP2[2], TRANS2[2] = registration_result(source_2, target, mean_normal2, ROT[2])
        ICP2[3], TRANS2[3] = registration_result(source_2, target, mean_normal2, ROT[3])
        ICP2[4], TRANS2[4] = registration_result(source_2, target, mean_normal2, ROT[4])
        ICP2[5], TRANS2[5] = registration_result(source_2, target, mean_normal2, ROT[5])

        RMSE1 = [ICP1[0].inlier_rmse, ICP1[1].inlier_rmse, ICP1[2].inlier_rmse, ICP1[3].inlier_rmse, ICP1[4].inlier_rmse, ICP1[5].inlier_rmse] 
        min_rmse1 = np.argmin(RMSE1)

        RMSE2 = [ICP2[0].inlier_rmse, ICP2[1].inlier_rmse, ICP2[2].inlier_rmse, ICP2[3].inlier_rmse, ICP2[4].inlier_rmse, ICP2[5].inlier_rmse] 
        min_rmse2 = np.argmin(RMSE2)

        if RMSE1[min_rmse1] > RMSE2[min_rmse2]:
            pre_trans = np.matmul(init_trans2, init_trans)
            trans_inter = np.matmul(ICP2[min_rmse2].transformation, Rt2T(ROT[min_rmse2], TRANS2[min_rmse2]))
            trans_comb = np.matmul(trans_inter, pre_trans)
        else:
            trans_inter = np.matmul(ICP1[min_rmse1].transformation, Rt2T(ROT[min_rmse1], TRANS1[min_rmse1]))
            trans_comb = np.matmul(trans_inter, init_trans) 
    except Exception:
        pass

    # copy_copy = copy.deepcopy(source)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # copy_copy.paint_uniform_color([1, 0, 0])
    # target.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([copy_copy.transform(trans_comb), target, mesh_frame], point_show_normal=False)

    return trans_comb


# Warm start after the initial rotation estimation
def RotEst_warmstart(source, target, prev_registration):
    try:
        source_temp = copy.deepcopy(source)
        source_temp.transform(prev_registration)

        result = refine_registration(source_temp, target, 1)
        trans_comb = np.matmul(result.transformation, prev_registration)
    except Exception:
        pass

    # copy_copy = copy.deepcopy(source)
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    # copy_copy.paint_uniform_color([1, 0, 0])
    # target.paint_uniform_color([0, 1, 0])
    # o3d.visualization.draw_geometries([copy_copy.transform(trans_comb), target, mesh_frame], point_show_normal=False)

    return trans_comb


# Evaluation whether registration is valid
def score(source, target):
    pts_source = np.asarray(source.points)
    pts_target = np.asarray(target.points)
    
    num_source = pts_source.shape[0]
    num_target = pts_target.shape[0]

    added = np.append(pts_target, pts_source[0]).reshape((-1, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(added)

    tick = np.zeros((num_target, 1))

    for i in range(num_source):
        pcd.points[num_target] = pts_source[i]
        target_tree = o3d.geometry.KDTreeFlann(pcd)
        [_, idx, _] = target_tree.search_knn_vector_3d(pcd.points[num_target], 2)
        tick[idx[1]] += 1

    score = np.count_nonzero(tick) / num_source

    return score


# Rotation matrix construction
def rotation_matrix(a, b):
    unit_vector1 = a / np.linalg.norm(a)
    unit_vector2 = b / np.linalg.norm(b)
    v = np.cross(unit_vector1, unit_vector2)
    c = np.dot(unit_vector1, unit_vector2)
    vx = [[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]]
    R = np.eye(3) + vx + np.dot(vx, vx)/(1+c)

    return R


# OBB veritice calculation
def get_vertice(edges):
    point = o3d.geometry.PointCloud()
    point.points = o3d.utility.Vector3dVector(edges)
    tree = o3d.geometry.KDTreeFlann(point)
    idx = tree.search_knn_vector_3d(point.points[0], 5)[1]
    long_vector = edges[0] - edges[idx[4]]
    small_vector = edges[0] - edges[idx[1]]

    return small_vector, long_vector

# ICP from Open3D
def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 5
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())

    return result


# Display registration result (between source and target)
def registration_result(source, target, arm, rotation):
    source_temp = copy.deepcopy(source)
    trans = np.dot(rotation, arm)
    source_temp.transform(Rt2T(rotation, trans))
    source_temp.translate(trans)
    result = refine_registration(source_temp, target, 0.1)

    return result, trans
    

# Form homogeneous transformation
def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t

    return T


# YOLACT++ :3,3
if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)

    if args.detect:
        cfg.eval_mask_branch = False

    with torch.no_grad():
        if not os.path.exists('results'):
            os.makedirs('results')

        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        print('Loading model...')
        net = Yolact()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate_ros(net)
