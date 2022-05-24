from zlib import ZLIB_RUNTIME_VERSION
import numpy as np

def fetch_source_dependent_params_dlf(model):
    def _get_layer_index(z, depth):
        depth_mat = depth.reshape(-1,1) @ np.ones_like([z])
        is_above = (depth_mat - z) < 0
        layer_index = np.count_nonzero(is_above, axis=0)
        return layer_index

    if model.source_name in {"VMD", "VED"}:
        xs, ys, zs = model.source_place[0]
        x, y, z = model.coordinate
        # horzontal distance between source & measurement point
        rho = ((xs-x) ** 2 + (ys-y) ** 2) ** 0.5
        if rho == 0:
            rho = 1e-4

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
        model.zs = zs
        model.z = z
        model.si, model.ri = si, ri
        model.rho = rho
        model.cos_phi, model.sin_phi = cos_phi, sin_phi
        model.divisor = rho

    elif model.source_name in {"HMD", "HED"}:
        psi = model.source.azmrad

        xs, ys, zs = model.source_place[0]
        x, y, z = model.coordinate

        # horzontal distance between source & measurement point
        rho = ((xs-x) ** 2 + (ys-y) ** 2) ** 0.5
        if rho == 0:
            rho = 1e-4

        #  rotate receive-point around source-point so that dipole is adjusted
        # co-axially with x-axis.
        xy_source = np.array([[xs], [ys]])
        xy_receive = np.array([[x], [y]])
        rvec = xy_receive - xy_source
        rot_mat = np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])
        rvec_rotated = rot_mat @ rvec
        xy_new = xy_source + rvec_rotated
        x_new = xy_new[0]
        y_new = xy_new[1]

        # sine & cosine of azimuth around the x-coaxial dipole
        cos_phi = (x_new - xs) / rho
        sin_phi = (y_new - ys) / rho

        # to avoid non-differential point
        if zs in model.depth:
            zs = zs - 1e-8
        if zs == z:
            zs = zs - 1e-8

        # in which layer source & measurement points exist
        si = _get_layer_index(zs, model.depth) # for source
        ri = _get_layer_index(z, model.depth) # for receiver

        # return values to model
        model.zs = zs
        model.z = z
        model.si, model.ri = si, ri
        model.rho = rho
        model.cos_phi, model.sin_phi = cos_phi, sin_phi
        # divisor
        model.divisor = rho

    elif model.source_name in {"AMD", "AED"}:

        psi = model.source.azmrad

        xs, ys, zs = model.source_place[0]
        x, y, z = model.coordinate

        # horzontal distance between source & measurement point
        rho = ((xs-x) ** 2 + (ys-y) ** 2) ** 0.5
        if rho == 0:
            rho = np.array([1e-4])
        
        #  rotate receive-point around source-point horizontally so that 
        # dipole is adjusted normal to y-axis
        xy_source = np.array([[xs], [ys]])
        xy_receive = np.array([[x], [y]])
        rvec = xy_receive - xy_source
        rot_mat = np.array([
            [np.cos(psi), np.sin(psi)],
            [-np.sin(psi), np.cos(psi)]
        ])
        rvec_rotated = rot_mat @ rvec
        xy_new = xy_source + rvec_rotated
        x_new = xy_new[0]
        y_new = xy_new[1]

        # for vertical dipole projection
        cos_phi_v = (x - xs) / rho
        sin_phi_v = (y - ys) / rho
        # for x-coaxial horizontal dipole projection
        cos_phi_h = (x_new - xs) / rho
        sin_phi_h = (y_new - ys) / rho

        # to avoid non-differential point
        if zs in model.depth:
            zs = zs - 1e-8
        if zs == z:
            zs = zs - 1e-8

        # in which layer source & measurement points exist
        si = _get_layer_index(zs, model.depth)
        ri = _get_layer_index(z, model.depth)

        # return values to model
        model.zs = zs
        model.z = z
        model.si, model.ri = si, ri
        model.rho = rho
        model.cos_phi_v, model.sin_phi_v = cos_phi_v, sin_phi_v
        model.cos_phi_h, model.sin_phi_h = cos_phi_h, sin_phi_h
        model.divisor = rho

    elif model.source_name == "CircularLoop":
        xs, ys, zs = model.source_place[0]
        x, y, z = model.coordinate
        radius = model.source.radius
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
        model.zs = zs
        model.z = z
        model.si, model.ri = si, ri
        model.rho = rho
        model.cos_phi, model.sin_phi = cos_phi, sin_phi
        model.divisor = radius

    elif model.source_name == "GroundedWire":
        xs, ys, zs = model.source_place.T
        x, y, z = model.coordinate
        length = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
        cos_theta = (xs[1] - xs[0]) / length
        sin_theta = (ys[1] - ys[0]) / length

        if (zs[0] != zs[1]):
            raise Exception('Z-coordinates of the wire must be the same value.')

        # 計算できない送受信座標が入力された場合の処理
        # 特異点の回避
        if zs[0] in model.depth:
            zs = zs - 1e-8
        if zs[0] == z:
            zs = zs - 1e-8

        if model.source.split is None:
            # 電線と受信点の最短距離の５分の1以下になるようなダイポール長を求める
            ssx = xs[1] - xs[0]
            ssy = ys[1] - ys[0]
            srx = x - xs[0]
            sry = y - ys[0]
            srz = z - zs[0]
            u_vec = np.array([ssx, ssy, 0])
            v_vec = np.array([srx, sry, srz])
            uv = np.dot(u_vec, v_vec)
            u2 = np.dot(u_vec, u_vec)
            t = uv/u2
            if t > 1:
                d_vec = v_vec - u_vec
                dist = np.sqrt(float(np.dot(d_vec, d_vec)))
            elif t < 0:
                dist = np.sqrt(float(np.dot(v_vec, v_vec)))
            else:
                d_vec = v_vec - t*u_vec
                dist = np.sqrt(float(np.dot(d_vec, d_vec)))
            sup = dist / 5
            split = int(length/sup) + 1
        else:
            split = int(model.source.split)

        # 節点
        xs_node = np.linspace(xs[0], xs[1], split + 1)
        ys_node = np.linspace(ys[0], ys[1], split + 1)
        zs_node = np.linspace(zs[0], zs[1], split + 1)

        dx = xs[1] - xs[0] / split
        dy = ys[1] - ys[0] / split
        dz = zs[1] - zs[0] / split
        ds = length / split

        xs_pole = xs_node[:-1] + dx / 2
        ys_pole = ys_node[:-1] + dy / 2
        zs_pole = zs_node[:-1] + dz / 2

        rot_matrix = np.array([
            [cos_theta, sin_theta, 0],
            [-sin_theta, cos_theta, 0],
            [0, 0, 1]
            ], dtype=float)

        rot_node_xyz = rot_matrix @ np.array([xs_node, ys_node, zs_node])
        rot_pole_xyz = rot_matrix @ np.array([xs_pole, ys_pole, zs_pole])
        rot_rec_xyz = np.dot(rot_matrix, np.array([x, y, z]))

        dist_x_pole = rot_rec_xyz[0] - rot_pole_xyz[0]
        dist_y_pole = rot_rec_xyz[1] - rot_pole_xyz[1]
        rho_pole = np.sqrt(dist_x_pole ** 2 + dist_y_pole ** 2)

        dist_x_node = rot_rec_xyz[0] - rot_node_xyz[0]
        dist_y_node = rot_rec_xyz[1] - rot_node_xyz[1]
        rho_node = np.sqrt(dist_x_node ** 2 + dist_y_node ** 2)
        rho_g_begin = rho_node[0]
        rho_g_end = rho_node[-1]

        rho = np.hstack([[rho_g_begin], rho_pole, [rho_g_end]])

        si = _get_layer_index(zs[0], model.depth)
        ri = _get_layer_index(z, model.depth)

        model.zs = zs[0]
        model.z = z
        model.dist_x_node, model.dist_y_node = dist_x_node, dist_y_node
        model.dist_x_pole, model.dist_y_pole = dist_x_pole, dist_y_pole
        model.rho = rho
        model.ds = ds
        model.si = si
        model.ri = ri
        model.split = split
        model.cos_theta = cos_theta
        model.sin_theta = sin_theta
        model.divisor = rho

    elif model.source_name == "PolygonalLoop":
        xs, ys, zs = model.source_place.T
        x, y, z = model.coordinate

        nvertex = len(xs)

        xs = np.append(xs, xs[0])
        ys = np.append(ys, ys[0])
        zs = np.append(zs, zs[0])

        length = np.sqrt((xs[1:] - xs[:-1]) ** 2 + (ys[1:] - ys[:-1]) ** 2)

        cos_theta = (xs[1:] - xs[:-1]) / length
        sin_theta = (ys[1:] - ys[:-1]) / length

        if np.any(zs != zs[0]):
            raise Exception('Z-coordinates of the wire must be the same value.')

        # 計算できない送受信座標が入力された場合の処理
        # 特異点の回避
        if zs[0] in model.depth:
            zs = zs - 1e-8
        if zs[0] == z:
            zs = zs - 1e-8
        
        if model.source.split is None:
            split = []
            for i in range(nvertex):
                # 電線と受信点の最短距離の５分の1以下になるようなダイポール長を求める
                ssx = xs[i+1] - xs[i]
                ssy = ys[i+1] - ys[i]
                srx = x - xs[i]
                sry = y - ys[i]
                srz = z - zs[i]
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
                split.append(int(length[i]/sup) + 1)
        else:
            split = model.source.split

        ds = length / split
        
        xs_pole_all = []
        ys_pole_all = []
        zs_pole_all = []
        rot_xs_pole_all = []
        rot_ys_pole_all = []
        rot_zs_pole_all = []
        dist_x_pole_all = []
        dist_y_pole_all = []
        rho_pole_all = []
        index_slice = [0]

        for i in range(nvertex):
            xs_node = np.linspace(xs[i], xs[i+1], split[i] + 1)
            ys_node = np.linspace(ys[i], ys[i+1], split[i] + 1)
            zs_node = np.linspace(zs[i], zs[i+1], split[i] + 1)

            dx = xs[i+1] - xs[i] / split[i]
            dy = ys[i+1] - ys[i] / split[i]
            dz = zs[i+1] - zs[i] / split[i]

            xs_pole = xs_node[:-1] + dx / 2
            ys_pole = ys_node[:-1] + dy / 2
            zs_pole = zs_node[:-1] + dz / 2

            xs_pole_all.append(xs_pole)
            ys_pole_all.append(ys_pole)
            zs_pole_all.append(zs_pole)

            rot_matrix = np.array([
                [cos_theta[i], sin_theta[i], 0],
                [-sin_theta[i], cos_theta[i], 0],
                [0, 0, 1]
                ], dtype=float)

            rot_pole_xyz = rot_matrix @ np.array([xs_pole, ys_pole, zs_pole])
            rot_rec_xyz = np.dot(rot_matrix, np.array([x, y, z]))

            rot_xs_pole_all.append(rot_pole_xyz[0])
            rot_ys_pole_all.append(rot_pole_xyz[1])
            rot_zs_pole_all.append(rot_pole_xyz[2])

            dist_x_pole = rot_rec_xyz[0] - rot_pole_xyz[0]
            dist_y_pole = rot_rec_xyz[1] - rot_pole_xyz[1]
            rho_pole = np.sqrt(dist_x_pole ** 2 + dist_y_pole ** 2)

            dist_x_pole_all.append(dist_x_pole)
            dist_y_pole_all.append(dist_y_pole)
            rho_pole_all.append(rho_pole)

        index_slice = np.append(0, np.cumsum(split))

        rho = np.hstack(rho_pole_all)

        si = _get_layer_index(zs[0], model.depth)
        ri = _get_layer_index(z, model.depth)

        model.zs = zs[0]
        model.z = z
        model.dist_x_pole, model.dist_y_pole = dist_x_pole_all, dist_y_pole_all
        model.rho = rho
        model.rho_list = rho_pole_all
        model.ds = ds
        model.si = si
        model.ri = ri
        model.nvertex = nvertex
        model.split = split
        model.index_slice = index_slice
        model.cos_theta = cos_theta
        model.sin_theta = sin_theta
        model.divisor = rho
    else:
        raise Exception