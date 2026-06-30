# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Utility functions related to point cloud
"""

import open3d as o3d
import numpy as np
import io
import struct

try:
    from pypcd import pypcd
except Exception:
    pypcd = None

def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points


def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, 4)

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lida data, shape: (n, 4)
    """

    lidar_xyz = lidar_data[:, :3].T
    # (3, n) -> (4, n), homogeneous transformation
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    lidar_int = lidar_data[:, 3]

    # transform to ego vehicle space, (3, n)
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :]
    # (n, 3)
    project_lidar_xyz = project_lidar_xyz.T
    # concatenate the intensity with xyz, (n, 4)
    projected_lidar = np.hstack((project_lidar_xyz,
                                 np.expand_dims(lidar_int, -1)))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list

def _pcd_numpy_dtype(fields, sizes, types, counts):
    type_map = {
        ("F", 4): "f4",
        ("F", 8): "f8",
        ("I", 1): "i1",
        ("I", 2): "i2",
        ("I", 4): "i4",
        ("I", 8): "i8",
        ("U", 1): "u1",
        ("U", 2): "u2",
        ("U", 4): "u4",
        ("U", 8): "u8",
    }
    dtype = []
    for field, size, type_, count in zip(fields, sizes, types, counts):
        np_type = type_map[(type_.upper(), int(size))]
        count = int(count)
        if count == 1:
            dtype.append((field, np_type))
        else:
            dtype.append((field, np_type, (count,)))
    return np.dtype(dtype)


def _lzf_decompress(data, expected_size):
    """
    Decompress PCL binary_compressed PCD payloads.

    PCD binary_compressed uses LZF. Keeping a small decoder here avoids adding
    a fragile pypcd/python-lzf dependency just for DAIR-V2X point clouds.
    """
    data = memoryview(data)
    out = bytearray()
    ip = 0
    data_len = len(data)

    while ip < data_len:
        ctrl = data[ip]
        ip += 1

        if ctrl < 32:
            length = ctrl + 1
            if ip + length > data_len:
                raise ValueError("Invalid LZF stream: literal overruns input")
            out.extend(data[ip:ip + length])
            ip += length
            continue

        length = ctrl >> 5
        ref_offset = (ctrl & 0x1f) << 8
        if length == 7:
            if ip >= data_len:
                raise ValueError("Invalid LZF stream: missing length byte")
            length += data[ip]
            ip += 1

        if ip >= data_len:
            raise ValueError("Invalid LZF stream: missing offset byte")
        ref_offset += data[ip]
        ip += 1

        ref_pos = len(out) - ref_offset - 1
        if ref_pos < 0:
            raise ValueError("Invalid LZF stream: bad back-reference")

        length += 2
        for _ in range(length):
            out.append(out[ref_pos])
            ref_pos += 1

    if len(out) != expected_size:
        raise ValueError(
            "Invalid LZF stream: expected %d bytes, got %d"
            % (expected_size, len(out))
        )
    return bytes(out)


def _read_binary_compressed_pcd(data_bytes, dtype, fields, points):
    if len(data_bytes) < 8:
        raise ValueError("Invalid binary_compressed PCD: missing size header")

    compressed_size, uncompressed_size = struct.unpack("<II", data_bytes[:8])
    compressed = data_bytes[8:8 + compressed_size]
    if len(compressed) != compressed_size:
        raise ValueError("Invalid binary_compressed PCD: truncated payload")

    raw = _lzf_decompress(compressed, uncompressed_size)
    arr = np.empty(points, dtype=dtype)

    offset = 0
    for field in fields:
        field_dtype = dtype.fields[field][0]
        nbytes = field_dtype.itemsize * points
        if offset + nbytes > len(raw):
            raise ValueError(
                "Invalid binary_compressed PCD: field %s overruns payload"
                % field
            )
        arr[field] = np.frombuffer(raw, dtype=field_dtype,
                                   count=points, offset=offset)
        offset += nbytes

    return arr


def _read_pcd_without_pypcd(pcd_path):
    header = {}
    header_lines = []
    with open(pcd_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PCD file: missing DATA line")
            decoded = line.decode("utf-8", errors="ignore").strip()
            header_lines.append(decoded)
            if decoded.startswith("#") or not decoded:
                continue
            key, *values = decoded.split()
            header[key.upper()] = values
            if key.upper() == "DATA":
                data_bytes = f.read()
                break

    fields = header["FIELDS"]
    sizes = [int(v) for v in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(v) for v in header.get("COUNT", ["1"] * len(fields))]
    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    data_kind = header["DATA"][0].lower()

    if data_kind == "ascii":
        arr = np.loadtxt(io.BytesIO(data_bytes), dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        columns = {name: arr[:, idx] for idx, name in enumerate(fields)}
    elif data_kind == "binary":
        dtype = _pcd_numpy_dtype(fields, sizes, types, counts)
        arr = np.frombuffer(data_bytes, dtype=dtype, count=points)
        columns = {name: arr[name] for name in fields}
    elif data_kind == "binary_compressed":
        dtype = _pcd_numpy_dtype(fields, sizes, types, counts)
        arr = _read_binary_compressed_pcd(data_bytes, dtype, fields, points)
        columns = {name: arr[name] for name in fields}
    else:
        raise ValueError("Unsupported PCD DATA format: %s" % data_kind)

    intensity = columns.get("intensity")
    if intensity is None:
        intensity = np.zeros_like(columns["x"], dtype=np.float32)
    else:
        intensity = intensity / 256.0

    pcd_np_points = np.stack(
        [columns["x"], columns["y"], columns["z"], intensity],
        axis=-1
    ).astype(np.float32)
    return pcd_np_points


def read_pcd(pcd_path):
    if pypcd is None:
        pcd_np_points = _read_pcd_without_pypcd(pcd_path)
        time = None
        del_index = np.where(np.isnan(pcd_np_points))[0]
        pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
        return pcd_np_points, time

    pcd = pypcd.PointCloud.from_path(pcd_path)
    time = None
    pcd_np_points = np.zeros((pcd.points, 4), dtype=np.float32)
    pcd_np_points[:, 0] = np.transpose(pcd.pc_data["x"])
    pcd_np_points[:, 1] = np.transpose(pcd.pc_data["y"])
    pcd_np_points[:, 2] = np.transpose(pcd.pc_data["z"])
    pcd_np_points[:, 3] = np.transpose(pcd.pc_data["intensity"]) / 256.0
    del_index = np.where(np.isnan(pcd_np_points))[0]
    pcd_np_points = np.delete(pcd_np_points, del_index, axis=0)
    return pcd_np_points, time
