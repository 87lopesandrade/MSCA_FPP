import numpy as np
import yaml
import matplotlib.pyplot as plt
import gc
from numpy import ndarray
import cv2
import open3d as o3d
from msca.config import (
    MODULATION_THRESHOLD_FACTOR, STD_MIN_THRESH, STD_MAX_THRESH,
    STATISTICAL_OUTLIER_NEIGHBORS, STATISTICAL_OUTLIER_STD_RATIO,
    RANSAC_DISTANCE_THRESHOLD, RANSAC_N, RANSAC_NUM_ITERATIONS
)

class InverseTriangulation:
    def __init__(self, calib_file):

        self.left_images = np.array([])
        self.right_images = np.array([])
        self.left_mask = np.array([])
        self.right_mask = np.array([])

        # Initialize all camera parameters in a single nested dictionary
        self.camera_params = {
            'left': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'right': {'kk': np.array([]), 'kc': np.array([]), 'r': np.array([]), 't': np.array([])},
            'stereo': {'R': np.array([]), 'T': np.array([])}
        }

        self.read_calibration_file(calib_file)

        self.z_scan_step = None
        self.num_points = None
        self.max_gpu_usage = self.set_datalimit() // 3

        # self.uv_left = []
        # self.uv_right = []

    def read_calibration_file(self, calib_file):
        """
        Lê o arquivo .npz gerado pelo calibration.py do workspace local.
        Assume a Câmera Esquerda como Origem (Mundo).
        """
        data = np.load(calib_file)
        
        # Câmera Esquerda (Origem)
        self.camera_params['left']['kk'] = data['mtx_l'].astype(np.float64)
        self.camera_params['left']['kc'] = data['dist_l'].astype(np.float64).flatten()
        self.camera_params['left']['r'] = np.eye(3, dtype=np.float64)
        self.camera_params['left']['t'] = np.zeros(3, dtype=np.float64)

        # Câmera Direita (relativa à esquerda)
        self.camera_params['right']['kk'] = data['mtx_r'].astype(np.float64)
        self.camera_params['right']['kc'] = data['dist_r'].astype(np.float64).flatten()
        self.camera_params['right']['r'] = data['R'].astype(np.float64)
        self.camera_params['right']['t'] = data['T'].astype(np.float64).flatten()

        self.camera_params['stereo']['R'] = data['R'].astype(np.float64)
        self.camera_params['stereo']['T'] = data['T'].astype(np.float64)

    def read_images(self, left_imgs, right_imgs, left_mask, right_mask):
        if len(left_imgs) != len(right_imgs):
            raise Exception("Number of images do not match")
        self.left_images = np.asarray(left_imgs)
        self.right_images = np.asarray(right_imgs)
        self.left_mask = np.asarray(left_mask)
        self.right_mask = np.asarray(right_mask)

    def points3d_zstep(self, x_lim=(-5, 5), y_lim=(-5, 5), xy_step=1.0, z_lin=np.arange(0, 100, 0.1), visualize=False):
        """
            Create a 3D space of combination from linear arrays of X Y Z
            Parameters:
                x_lim: Begin and end of linear space of X
                y_lim: Begin and end of linear space of Y
                z_lin: numpy array of z to be tested
                xy_step: Step size between X and Y
                visualize: Visualize the 3D space
            Returns:
                cube_points: combination of X Y and Z
        """
        x_lin = np.arange(x_lim[0], x_lim[1], xy_step)
        y_lin = np.arange(y_lim[0], y_lim[1], xy_step)

        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')

        c_points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=c_points[:, 0], y=c_points[:, 1], z=c_points[:, 2])

        self.num_points = c_points.shape[0]
        self.z_scan_step = np.unique(c_points[:, 2]).shape[0]

        return c_points

    def points3D_arrays(self, x_lin: ndarray, y_lin: ndarray, z_lin: ndarray, visualize: bool = True) -> ndarray:
        """
        Crete 3D meshgrid of points based on input vectors of x, y and z
        :param x_lin: linear space of x points
        :param y_lin: linear space of y points
        :param z_lin: linear space of z points
        :param visualize: If true plot a 3d graph of points
        :return: 3D meshgrid points size (N,3) where N = len(x)*len(y)*len(z)
        """
        mg1, mg2, mg3 = np.meshgrid(x_lin, y_lin, z_lin, indexing='ij')
        points = np.stack([mg1, mg2, mg3], axis=-1).reshape(-1, 3)

        if visualize:
            self.plot_3d_points(x=points[:, 0], y=points[:, 1], z=points[:, 2])

        self.num_points = points.shape[0]
        self.z_scan_step = np.unique(points[:, 2]).shape[0]

        return np.asarray(points)

    def plot_3d_points(self, x, y, z, color=None, title='Plot 3D of max correlation points'):
        """
        Plot 3D points as scatter points where color is based on Z value
        Parameters:
            x: array of x positions
            y: array of y positions
            z: array of z positions
            color: Vector of point intensity grayscale
        """

        if color is None:
            color = z
        cmap = 'viridis'
        # Plot the 3D scatter plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.title.set_text(title)

        scatter = ax.scatter(x, y, z, c=color, cmap=cmap, marker='o')
        # ax.set_zlim(0, np.max(z))
        colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label('Z Value Gradient')

        # Add labels
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def save_points(self, data, filename, delimiter=','):
        """
        Save a 2D NumPy array to a CSV file.

        :param array: 2D numpy array
        :param filename: Output CSV filename
        """
        # Save the 2D array as a CSV file
        np.savetxt(filename, data, delimiter=delimiter)
        print(f"Array saved to {filename}")

    def set_datalimit(self):
        """
        Identify memory limit for CPU RAM
        """
        # Fixando limite seguro de 4GB para processamento em batch no Mac
        return 4.0



    def remove_img_distortion(self, img, camera):
        return cv2.undistort(img, self.camera_params[camera]['kk'], self.camera_params[camera]['kc'])

    def transform_gcs2ccs(self, points_3d, cam_name, undist=False):
        """
        Transform Global Coordinate System (xg, yg, zg)
        to Camera's Coordinate System (xc, yc, zc) and transform to Image's plane (uv)
        Returns:
            uv_image_points: (2,N) reprojected points to image's plane
        """
        # Convert all inputs to CuPy arrays for GPU computation
        xyz_gcs = np.asarray(points_3d)
        k = np.asarray(self.camera_params[cam_name]['kk'])
        dist = np.asarray(self.camera_params[cam_name]['kc'])
        rot = np.asarray(self.camera_params[cam_name]['r'])
        tran = np.asarray(self.camera_params[cam_name]['t'])

        # Estimate memory required for processing
        bytes_per_float32 = 8
        memory_per_point = (4 * 3 * bytes_per_float32) + (3 * bytes_per_float32)
        total_memory_required = self.num_points * memory_per_point

        # Adjust batch size based on memory limitations
        if total_memory_required > self.max_gpu_usage * 1024 ** 3:
            points_per_batch = int(
                (self.max_gpu_usage * 1024 ** 3 // memory_per_point) // 10)  # Reduce batch size more aggressively
            # print(f"Processing {points_per_batch} points per batch due to memory limitations.")
        else:
            points_per_batch = self.num_points  # Process all points at once

        # Initialize a list to store results on the GPU
        uv_points_list = np.empty((2, xyz_gcs.shape[0]), dtype=np.float32)

        # Process points in batches
        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
            xyz_gcs_batch = xyz_gcs[i:end]

            # Add one extra line of ones to the global coordinates
            ones = np.ones((xyz_gcs_batch.shape[0], 1), dtype=np.float32)
            xyz_gcs_1 = np.hstack((xyz_gcs_batch, ones))

            # Create the rotation and translation matrix
            rt_matrix = np.vstack(
                (np.hstack((rot, tran[:, None])), np.array([0, 0, 0, 1], dtype=np.float32))
            )

            # Multiply the RT matrix with global points [X; Y; Z; 1]
            xyz_ccs = np.dot(rt_matrix, xyz_gcs_1.T)
            del xyz_gcs_1

            # Normalize by dividing by Z to get normalized image coordinates
            epsilon = 1e-10  # Small value to prevent division by zero
            xyz_ccs_norm = np.hstack(
                (xyz_ccs[:2, :].T / np.maximum(xyz_ccs[2, :, np.newaxis], epsilon),
                 np.ones((xyz_ccs.shape[1], 1), dtype=np.float32))
            ).T
            del xyz_ccs

            # Apply distortion using the GPU
            if undist:
                xyz_ccs_norm_dist = self.undistorted_points(xyz_ccs_norm.T, dist)
                del xyz_ccs_norm  # Free memory
                uv_points_batch = np.dot(k, xyz_ccs_norm_dist.T).astype(np.float32)
                del xyz_ccs_norm_dist  # Free memory
            else:
                # Compute image points using the intrinsic matrix K
                uv_points_batch = np.dot(k, xyz_ccs_norm).astype(np.float32)
                del xyz_ccs_norm  # Free memory

            # Transfer results back to CPU after processing each batch
            uv_points_list[:, i:end] = uv_points_batch[:2, :]

            # Free GPU memory after processing each batch
            
            gc.collect()

        # Transfer final result to CPU in a single operation
        return uv_points_list

    def undistorted_points(self, points, dist):
        """
        GPU version of the undistorted points function using CuPy.
        Applies radial and tangential distortion.

        Parameters:
            points: 2D array of normalized image coordinates [x, y].
            dist: Distortion coefficients [k1, k2, p1, p2, k3].

        Returns:
            Distorted points on the GPU.
        """
        # Extract distortion coefficients
        k1, k2, p1, p2, k3 = dist

        # Split points into x and y coordinates
        x, y = points[:, 0], points[:, 1]

        # Calculate r^2 (squared distance from the origin)
        r2 = x ** 2 + y ** 2

        # Radial distortion
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3

        # Tangential distortion
        x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2)
        y_tangential = p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y

        # Compute distorted coordinates
        x_distorted = x * radial + x_tangential
        y_distorted = y * radial + y_tangential

        # Stack the distorted points
        distorted_points = np.vstack([x_distorted, y_distorted, np.ones_like(x)]).T

        # Clean up intermediate variables to free memory
        del x, y, r2, radial, x_tangential, y_tangential
        

        return distorted_points

    def bi_interpolation(self, images, modulation_map, uv_points, window_size=3):
        """
        Perform bilinear interpolation on a stack of images at specified uv_points on the GPU.

        Parameters:
        ----------
        images : (height, width, num_images) array or (height, width) for a single image.
        uv_points : (2, N) array of UV points where N is the number of points.
        window_size : int
            Unused here but kept for consistency with spatial functions.

        Returns:
        -------
        interpolated_cpu : np.ndarray
            Interpolated pixel values for each point.
        std_cpu : np.ndarray
            Standard deviation of the corner pixels used for interpolation.
        """
        images = np.asarray(images)
        uv_points = np.asarray(uv_points)
        mod_threshold = 0.01 * np.max(modulation_map)
        mod_threshold = np.float64(mod_threshold)

        if len(images.shape) == 2:  # Convert single image to a stack with one image
            images = images[:, :, np.newaxis]

        height, width, num_images = images.shape

        # Estimate memory usage per point
        memory_per_point = 4 * num_images * 4
        points_per_batch = max(1, int(self.max_gpu_usage * 1024 ** 3 // memory_per_point))

        # Output arrays on GPU
        interpolated = np.zeros((self.num_points, num_images), dtype=np.float32)
        std = np.zeros((self.num_points, num_images), dtype=np.float32)

        for i in range(0, self.num_points, points_per_batch):
            end = min(i + points_per_batch, self.num_points)
            uv_batch = uv_points[:, i:end]

            # Compute integer and fractional parts of UV coordinates
            x = uv_batch[0].astype(np.float32)
            y = uv_batch[1].astype(np.float32)

            x1 = np.clip(np.floor(x).astype(np.int32), 0, width - 1)
            y1 = np.clip(np.floor(y).astype(np.int32), 0, height - 1)
            x2 = np.clip(x1 + 1, 0, width - 1)
            y2 = np.clip(y1 + 1, 0, height - 1)

            x_diff = x - x1
            y_diff = y - y1
            for k in range(num_images):
                # Vectorized extraction of corner pixels
                p11 = images[y1, x1, k]  # Top-left
                p12 = images[y2, x1, k]  # Bottom-left
                p21 = images[y1, x2, k]  # Top-right
                p22 = images[y2, x2, k]  # Bottom-right

                mod_p11 = modulation_map[y1, x1]
                mod_p12 = modulation_map[y2, x1]
                mod_p21 = modulation_map[y1, x2]
                mod_p22 = modulation_map[y2, x2]

                # Check if all corner modulations are above the threshold - Remove points with less than 1% of the modulation map value
                if np.all(np.array([mod_p11, mod_p12, mod_p21, mod_p22]) < mod_threshold):
                    # If any modulation is below the threshold, discard or adjust interpolation
                    interpolated[i:end, k] = np.nan # You can replace with NaN or other value
                    std[i:end, k] = np.nan  # Reset the standard deviation too
                else:
                    # Bilinear interpolation
                    interpolated_batch = (
                            p11 * (1 - x_diff) * (1 - y_diff) +
                            p21 * x_diff * (1 - y_diff) +
                            p12 * (1 - x_diff) * y_diff +
                            p22 * x_diff * y_diff)

                    std_batch = np.std(np.vstack([p11, p12, p21, p22]), axis=0)

                    # Store results in GPU arrays
                    interpolated[i:end, k] = interpolated_batch
                    std[i:end, k] = std_batch

                    del p11, p12, p21, p22, std_batch, interpolated_batch

                
                gc.collect()
        # Convert results to CPU
        # interpolated_cpu = np.asarray(interpolated_gpu)
        # std_cpu = np.asarray(std_gpu)

        if images.shape[2] == 1:  # Flatten output for single image
            interpolated_cpu = interpolated[:, 0]
            std_cpu = std[:, 0]

        return interpolated, std

    def phase_map(self, interp_left, interp_right, debug=False):
        """
        Identify minimum phase map value.
        Parameters:
            interp_left: left interpolated points (1D array, cupy.ndarray)
            interp_right: right interpolated points (1D array, cupy.ndarray)
            debug: if true, visualize phi_map array
        Returns:
            phi_min_id: indices of minimum phase map values (cupy.ndarray).
        """

        # Compute the absolute difference between left and right interpolations
        diff_phi = np.abs(interp_left - interp_right)

        # Reshape the array for efficient block processing
        diff_phi_blocks = diff_phi.reshape(-1, self.z_scan_step)

        # Find the indices of minimum values within each block
        block_min_indices = np.argmin(diff_phi_blocks, axis=1)

        # Adjust indices to account for the block position
        phi_min_id = block_min_indices + np.arange(len(block_min_indices)) * self.z_scan_step

        # Debug: Visualize the phase map if requested
        if debug:
            # Transfer data back to NumPy for plotting
            diff_phi_blocks_np = np.asarray(diff_phi_blocks)
            plt.figure()
            plt.plot(diff_phi_blocks_np.T)
            plt.title("Phase Difference Map")
            plt.xlabel("Index within block")
            plt.ylabel("Phase Difference")
            plt.show()

        return phi_min_id

    def fringe_masks(self, uv_l, uv_r, std_l, std_r, phi_id, min_thresh=STD_MIN_THRESH, max_thresh=STD_MAX_THRESH, window_size=3):
        """
        Mask from fringe process to remove outbounds points.
        Paramenters:
            std_l: STD interpolation image's points
            std_r: STD interpolation image's points
            phi_id: Indices for min phase difference
            min_thresh: max threshold for STD
            max_thresh: min threshold for STD
        Returns:
             valid_mask: Valid 3D points on image's plane
        """

        # converte as coordenadas em um array cupy
        uv_l = np.asarray(uv_l)
        uv_r = np.asarray(uv_r)

        # Verifica se as coordenadas estão dentro dos limites das máscaras
        valid_u_l = (uv_l[0, :] >= 0) & (uv_l[0, :] < self.left_mask.shape[1])
        valid_v_l = (uv_l[1, :] >= 0) & (uv_l[1, :] < self.left_mask.shape[0])
        valid_u_r = (uv_r[0, :] >= 0) & (uv_r[0, :] < self.right_mask.shape[1])
        valid_v_r = (uv_r[1, :] >= 0) & (uv_r[1, :] < self.right_mask.shape[0])

        # Aplica as verificações de validade nas coordenadas UV para evitar indexação fora do limite
        valid_uv_l = valid_u_l & valid_v_l
        valid_uv_r = valid_u_r & valid_v_r

        # Verifica os pontos válidos nas máscaras (aplica as coordenadas para obter as máscaras) - usando a média em vez do máximo para ser mais robusto a pontos de saturação especular
        mod_thresh_l = MODULATION_THRESHOLD_FACTOR * np.mean(self.left_mask)
        mod_thresh_r = MODULATION_THRESHOLD_FACTOR * np.mean(self.right_mask)
        valid_uv_l &= (self.left_mask[uv_l[1, :].clip(0, self.left_mask.shape[0] - 1).astype(int), uv_l[0, :].clip(0, self.left_mask.shape[1] - 1).astype(int)] > mod_thresh_l)
        valid_uv_r &= (self.right_mask[uv_r[1, :].clip(0, self.right_mask.shape[0] - 1).astype(int), uv_r[0, :].clip(0, self.right_mask.shape[1] - 1).astype(int)] > mod_thresh_r)

        # Combine as verificações dos limites
        valid_uv = valid_uv_r & valid_uv_l

        # Máscara para `phi_id`
        phi_mask = np.zeros(uv_l.shape[1], dtype=bool)
        phi_mask[phi_id] = True

        # Verificação dos thresholds de `std` para pontos válidos
        valid_l = (min_thresh < std_l) & (std_l < max_thresh)
        valid_r = (min_thresh < std_r) & (std_r < max_thresh)
        valid_std = valid_r[:, 0] & valid_l[:, 0]

        std_mod_l = np.std(self.left_mask)
        std_mod_r = np.std(self.right_mask)

        mod_threshold_l = 0.3 * np.max(self.left_mask)
        mod_threshold_r = 0.3 * np.max(self.right_mask)

        valid_mod_r = (0 < std_mod_r) & (std_mod_r < mod_threshold_r)
        
        # Desabilitando valid_mod global pois ele pode falhar dependendo da iluminação da cena 
        mask = valid_uv & phi_mask & valid_std

        return mask

    def fringe_process(self, points_3d: ndarray, save_points: bool = True, visualize: bool = False) -> ndarray:
        """
        Zscan for stereo fringe process
        Parameters:
            save_points: boolean to save or not image
            visualize: boolean to visualize result
        :return:
            measured_pts: Valid 3D global coordinate points
        """
        if points_3d.shape[0] == 0:
            return np.array([])
        self.num_points = points_3d.shape[0]

        # Converte as coordenadas 3D dos pontos para as coordenadas de imagem da câmera esquerda e direita
        uv_left = self.transform_gcs2ccs(points_3d, cam_name='left')
        uv_right = self.transform_gcs2ccs(points_3d, cam_name='right')

        # Realiza a interpolação bicúbica nas imagens das câmeras, retorna o valor interpolado e o desvio padrão da interpolação
        inter_left, std_left = self.bi_interpolation(self.left_images, self.left_mask,uv_left)
        inter_right, std_right = self.bi_interpolation(self.right_images, self.right_mask,uv_right)

        # Cálcula o índice da fase mínima entre s valores interpoados
        phi_min_id = self.phase_map(inter_left, inter_right)

        # Filra os pontos 3D com base no mapa de modularização, os desvios padrões da interpolação e os valores mínimos da fase
        measured_pts = points_3d[self.fringe_masks(uv_l = uv_left, uv_r = uv_right, std_l = std_left, std_r = std_right, phi_id = phi_min_id)]

        if save_points:
            self.save_points(measured_pts, filename='./fringe_points_results_calota.txt')

        if visualize:
            self.plot_3d_points(measured_pts[:, 0], measured_pts[:, 1], measured_pts[:, 2], color=None,
                                title="Fringe process output points")

        # deleta as variáveis temporárias para liberar memória
        del uv_left, uv_right,inter_left, inter_right, std_left, std_right, phi_min_id
        return measured_pts

    def filter_points_by_depth(self, points, depth_threshold=0.05, octree_depth=30, std_ratio=STATISTICAL_OUTLIER_STD_RATIO, nb_neighbors=STATISTICAL_OUTLIER_NEIGHBORS):
        if len(points) == 0:
            return points
        import open3d as o3d
        """
        Filtra pontos de uma nuvem de pontos 3D com base na profundidade média dentro de blocos da Octree.

        Parameters:
            points (np.ndarray or np.ndarray): Pontos da nuvem de pontos.
            depth_threshold (float): Tolerância para variação em relação à profundidade média.
            octree_depth (int): Profundidade máxima da Octree.
            std_ratio (float): Razão de desvio padrão para o filtro estatístico.
            nb_neighbors (int): Número de vizinhos para considerar no filtro estatístico.

        Returns:
            filtered_pcd (open3d.geometry.PointCloud): Nuvem de pontos filtrada.
        """

        # Converte para NumPy, se necessário
        if isinstance(points, np.ndarray):
            points = np.asarray(points)

        # Cria a PointCloud no Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Bypass the destructive octree depth filter entirely
        filtered_points = points

        # Converte pontos filtrados para Open3D
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(np.array(filtered_points))

        # Filtro estatístico para remover outliers de acordo com número de pixels vizinhos e distãncia
        filtered_pcd, _ = filtered_pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

        # ----- Alinhamento do Plano Base -----
        # Como o referencial é a Câmera Esquerda, a peça fica inclinada.
        # Usa RANSAC para achar o plano base e rotacionar a nuvem para o plano XY (Z=0).
        if len(filtered_pcd.points) >= 3:
            plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=RANSAC_DISTANCE_THRESHOLD, ransac_n=RANSAC_N, num_iterations=RANSAC_NUM_ITERATIONS)
            a, b, c, d = plane_model
            n = np.array([a, b, c])
            n = n / np.linalg.norm(n)
            if c < 0:
                n = -n
            
            k = np.array([0.0, 0.0, 1.0])
            v = np.cross(n, k)
            s = np.linalg.norm(v)
            c_val = np.dot(n, k)
            
            if s != 0:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_val) / (s ** 2))
            else:
                R = np.eye(3)
                
            filtered_pcd.rotate(R, center=(0,0,0))
            
            # Ajustar Z para que o plano base fique em Z=0
            points_aligned = np.asarray(filtered_pcd.points)
            points_inliers = points_aligned[inliers]
            if len(points_inliers) > 0:
                mean_z = np.mean(points_inliers[:, 2])
                points_aligned[:, 2] -= mean_z
                filtered_pcd.points = o3d.utility.Vector3dVector(points_aligned)

        return np.asarray(filtered_pcd.points)