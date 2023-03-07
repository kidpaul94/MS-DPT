import cv2 
import numpy as np
import open3d as o3d

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

# Form homogeneous transformation
def Rt2T(R,t):
    T = np.identity(4)
    T[:3,:3] = R
    T[:3,3] = t

    return T

def gen_dict(path='./models/rotary_arm.ply'):
    # Generate CAD point cloud
    mesh = o3d.io.read_triangle_mesh(path)
    target = mesh.sample_points_uniformly(number_of_points = 2048)
    save_normal = target.normals
    pts_target = np.asarray(target.points)
    zero_mean = pts_target - pts_target.mean(axis=0, keepdims=True)
    max_dist = max(np.linalg.norm(zero_mean, axis=1))
    target_norm = zero_mean / max_dist

    target.points = o3d.utility.Vector3dVector(target_norm)
    target.normals = save_normal

    # Calculate OBB and its vertices
    obb = target.get_oriented_bounding_box()
    edge_target = np.asarray(obb.get_box_points())
    rotax, target_vertice = get_vertice(edge_target)

    return target, obb, max_dist, rotax, target_vertice

# Visualize object pose on RGB images
def vispose(image, mask, RT):
    try:
        solutions = np.argwhere(mask[0] != 0)
        mean_x = int(np.mean(solutions[:,1]))
        mean_y = int(np.mean(solutions[:,0]))
        mean = (mean_x, mean_y)

        axis = np.int16([[40,0,0], [0,40,0], [0,0,40]])
        projection = np.dot(RT[:3,:3], axis)[:2,:3]
        projection[0,:3] = projection[0,:3] + mean_x
        projection[1,:3] = projection[1,:3] + mean_y
        projection = projection.astype(int)

        # project 3D points to image plane
        image = cv2.line(image, mean, tuple(projection[:2,0].ravel()), (0,0,255), 2)
        image = cv2.line(image, mean, tuple(projection[:2,1].ravel()), (0,255,0), 2)
        image = cv2.line(image, mean, tuple(projection[:2,2].ravel()), (255,0,0), 2)

        return image

    except Exception:
        pass