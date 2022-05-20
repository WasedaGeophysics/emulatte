import numpy as np

def organize(model):
    def _get_layer_index(z, depth):
        depth_mat = depth.reshape(-1,1) @ np.ones_like([z])
        is_above = (depth_mat - z) < 0
        layer_index = np.count_nonzero(is_above, axis=0)
        return layer_index

    vpts_source = ["VMD", "VED", "CircularLoop"]
    hpts_source = ["HMD", "HED"]

    if model.source_name in vpts_source:
        xs, ys, zs = model.source_place.T
        x, y, z = model.coordinate
        # horzontal distance between source & measurement point
        rho = ((xs-x) ** 2 + (ys-y) ** 2) ** 0.5
        if rho == 0:
            rho = 1e-8

        # azimuth of measurement point from source
        cos_phi = (x - xs) / rho
        sin_phi = (y - ys) / rho

        # to avoid non-differential point
        if zs in model.depth:
            zs = zs - 1e-8
        if zs == z:
            zs = zs - 1e-8

        # in which layer source & measurement points exist
        si = _get_layer_index(zs, model.depth)
        ri = _get_layer_index(z, model.depth)

        # return values to model
        model.xs, model.ys, model.zs = xs, ys, zs
        model.x, model.y, model.z = x, y, z
        model.si, model.ri = si, ri
        model.rho, model.cos_phi, model.sin_phi = rho, cos_phi, sin_phi

    elif model.source_name in hpts_source:
        xs, ys, zs = model.sc
        x, y, z = model.rc
        # horzontal distance between source & measurement point
        rho = ((xs-x) ** 2 + (ys-y) ** 2) ** 0.5
        if rho == 0:
            rho = 1e-8

        # azimuth of measurement point from source
        cos_phi = (x - xs) / rho
        sin_phi = (y - ys) / rho

        # to avoid non-differential point
        if zs in model.depth:
            zs = zs - 1e-8
        if zs == z:
            zs = zs - 1e-8

        # in which layer source & measurement points exist
        si_list = _get_layer_index(zs, model.depth) # for source
        ri = _get_layer_index(z, model.depth) # for receiver

        # return values to model
        model.xs, model.ys, model.zs = xs, ys, zs
        model.x, model.y, model.z = x, y, z
        model.si_list, model.ri = si_list, ri
        model.rho, model.cos_phi, model.sin_phi = rho, cos_phi, sin_phi

    elif model.source_type == "GroundedWire":
        xs, ys = model.sc.T
        x, y, z = model.rc
        zs = 1e-8
        length = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
        cos_theta = (xs[1] - xs[0]) / length
        sin_theta = (ys[1] - ys[0]) / length

        if (zs[0] != zs[1]) | (len()):
            raise Exception('Z-coordinates of the wire ends must be the same value.')

        # 計算できない送受信座標が入力された場合の処理
        # 特異点の回避
        if model.hankel_filter == "anderson801":
            delta_z = 1e-4
        else:
            delta_z = 1e-8
        if zs[0] in model.depth:
            zs = zs - delta_z
        if zs[0] == z:
            zs = zs - delta_z

        if model.source.split is None:
            ssx = xs[1] - xs[0]
            ssy = ys[1] - ys[0]
            srx = x - xs[0]
            sry = y - ys[0]
            srz = z - zs[0]
            u_vec = np.array([ssx, ssy, 0])
            v_vec = np.array([srx, sry, srz])
            uv = u_vec @ v_vec
            u2 = u_vec @ u_vec
            t = uv/u2
            if t > 1:
                d_vec = v_vec - u_vec
                dist = np.sqrt(float(d_vec @ d_vec))
            elif t < 0:
                dist = np.sqrt(float(v_vec @ v_vec))
            else:
                d_vec = v_vec - t*u_vec
                dist = np.sqrt(float(d_vec @ d_vec))
            sup = dist / 5
            split = int(length/sup) + 1
        else:
            split = model.source.split

        # 節点
        sx_node = np.linspace(xs[0], xs[1], split + 1)
        sy_node = np.linspace(ys[0], ys[1], split + 1)
        sz_node = np.linspace(zs[0], zs[1], split + 1)

        dx = xs[1] - xs[0] / split
        dy = ys[1] - ys[0] / split
        dz = zs[1] - zs[0] / split
        ds = length / split

        sx_dipole = sx_node[:-1] + dx / 2
        sy_dipole = sy_node[:-1] + dy / 2
        sz_dipole = sz_node[:-1] + dz / 2

        rot_matrix = np.array([
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [0, 0, 1]
            ], dtype=float)
        rot_sc = np.dot(rot_matrix, np.array([sx_dipole, sy_dipole, sz_dipole]).reshape(3,-1)).T
        rot_rc = np.dot(rot_matrix, model.rc)

        xx = rot_rc[0] - rot_sc[:, 0]
        yy = rot_rc[1] - rot_sc[:, 1]
        squared = (xx ** 2 + yy ** 2).astype(float)
        rn = np.sqrt(squared)

        si_list = _get_layer_index(zs[0], model.depth)
        ri = _get_layer_index(z, model.depth)

        model.xs, model.ys, model.zs = xs, ys, zs[0]
        model.x, model.y, model.z = x, y, z
        model.xx, model.yy, model.rn = xx, yy, rn
        model.ds = ds
        model.si_list = si_list
        model.ri = ri
        model.split = split
        model.cos_theta = cos_theta
        model.sin_theta = sin_theta

    elif model.source_name == "EmbededWire":
        pass
    elif model.source_name == "PolygonalLoop":
        pass
    else:
        raise Exception