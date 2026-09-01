import numpy as np
import cv2
import glob
import os
import json
import argparse
import matplotlib.pyplot as plt

def create_charuco_board(dictionary_id, squares_x, squares_y, square_length, marker_length):
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, 'CharucoBoard'):
            board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
            return dictionary, board
        elif hasattr(cv2.aruco, 'CharucoBoard_create'):
            board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_length, marker_length, dictionary)
            return dictionary, board
        else:
            raise AttributeError("Módulo Aruco incompatível ou não encontrado.")
    except Exception as e:
        print(f"Erro ao criar ChArUco board: {e}")
        return None, None

def detect_charuco_corners(img_gray, dictionary, board):
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        marker_corners, marker_ids, _ = detector.detectMarkers(img_gray)
        if marker_ids is not None and len(marker_ids) > 0:
            charuco_detector = cv2.aruco.CharucoDetector(board)
            charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img_gray)
            return charuco_corners, charuco_ids
    else:
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(img_gray, dictionary)
        if marker_ids is not None and len(marker_ids) > 0:
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img_gray, board)
            if retval > 0:
                return charuco_corners, charuco_ids
    return None, None

def filter_common_corners(corners_l, ids_l, corners_r, ids_r):
    if ids_l is None or ids_r is None:
        return None, None, None
    common_ids = np.intersect1d(ids_l, ids_r)
    if len(common_ids) < 6:
        return None, None, None

    common_corners_l = []
    common_corners_r = []
    valid_ids = []
    for cid in common_ids:
        idx_l = np.where(ids_l == cid)[0][0]
        idx_r = np.where(ids_r == cid)[0][0]
        common_corners_l.append(corners_l[idx_l])
        common_corners_r.append(corners_r[idx_r])
        valid_ids.append(cid)
    return np.array(common_corners_l).reshape(-1, 1, 2), np.array(common_corners_r).reshape(-1, 1, 2), np.array(valid_ids).reshape(-1, 1)

def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    mean_errors = []
    all_point_errors = []
    for i in range(len(objpoints)):
        imgpoints_projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        diff = imgpoints[i] - imgpoints_projected
        pt_errors = np.linalg.norm(diff, axis=-1).flatten()
        for j in range(len(pt_errors)):
            x, y = imgpoints[i][j][0]
            all_point_errors.append((x, y, pt_errors[j]))
        mean_errors.append(np.mean(pt_errors))
    return mean_errors, all_point_errors

def compute_epipolar_y_error(corners_l_list, corners_r_list, mtx_l, dist_l, mtx_r, dist_r, R1, P1, R2, P2):
    y_errors = []
    for i in range(len(corners_l_list)):
        pts_l = corners_l_list[i]
        pts_r = corners_r_list[i]
        rect_l = cv2.undistortPoints(pts_l, mtx_l, dist_l, R=R1, P=P1)
        rect_r = cv2.undistortPoints(pts_r, mtx_r, dist_r, R=R2, P=P2)
        y_l = rect_l[:, 0, 1]
        y_r = rect_r[:, 0, 1]
        errs = np.abs(y_l - y_r)
        y_errors.extend(errs.tolist())
    return y_errors

def calculate_fov_coverage(point_errors, img_size):
    grid_size = 20
    w, h = img_size
    pts = np.array([[p[0], p[1]] for p in point_errors])
    if len(pts) == 0:
        return 0.0
    H, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[grid_size, grid_size], range=[[0, w], [0, h]])
    covered_bins = np.count_nonzero(H)
    return (covered_bins / (grid_size**2)) * 100.0

def compute_vergence_and_resolution(R, T, mtx_l, working_dist_mm):
    # Vergência (Ângulo em Y usando Decomposição RQ)
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    vergence_angle = abs(euler_angles[1]) # Eixo Y
    
    # Resolução Z Teórica (para delta d = 0.1 px)
    fx = mtx_l[0, 0]
    fy = mtx_l[1, 1]
    f = (fx + fy) / 2.0
    b = np.linalg.norm(T)
    delta_z = (working_dist_mm ** 2) / (f * b) * 0.1
    return vergence_angle, delta_z

def plot_radial_distortion(ax, mtx_l, dist_l, mtx_r, dist_r, img_size):
    w, h = img_size
    max_radius = np.sqrt((w/2)**2 + (h/2)**2)
    f_l = (mtx_l[0,0] + mtx_l[1,1]) / 2.0
    
    r_norm = np.linspace(0, max_radius / f_l, 100)
    def calc_dist(k, r):
        k1, k2 = k[0], k[1]
        k3 = k[4] if len(k) >= 5 else 0.0
        return r * (k1 * r**2 + k2 * r**4 + k3 * r**6)
        
    delta_px_l = calc_dist(dist_l.flatten(), r_norm) * f_l
    delta_px_r = calc_dist(dist_r.flatten(), r_norm) * f_l
    r_px = r_norm * f_l
    
    ax.plot(r_px, delta_px_l, 'b-', label='Left Camera')
    ax.plot(r_px, delta_px_r, 'g-', label='Right Camera')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_title('5. Perfil de Distorção Radial')
    ax.set_xlabel('Distância do Centro Óptico (px)')
    ax.set_ylabel('Deslocamento (px)')
    ax.legend()
    ax.grid(True)

def generate_diagnostic_plots(errors_per_pair, y_errors, point_errors_l, img_size, objpoints, rvecs_l, tvecs_l, R, T, mtx_l, dist_l, mtx_r, dist_r, output_file="calibration_diagnostics.png"):
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Scatter Espacial 2D
    ax1 = fig.add_subplot(2, 3, 1)
    pts = np.array(point_errors_l)
    if len(pts) > 0:
        sc = ax1.scatter(pts[:, 0], pts[:, 1], c=pts[:, 2], cmap='jet', s=10, alpha=0.7)
        plt.colorbar(sc, ax=ax1, label='Erro (px)')
    ax1.set_xlim(0, img_size[0])
    ax1.set_ylim(img_size[1], 0)
    ax1.set_title("1. Distribuição Espacial (Left)")
    
    # Plot 2: Histograma do Y-Error
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(y_errors, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(y_errors), color='red', linestyle='dashed', linewidth=2, label=f"Média: {np.mean(y_errors):.3f}")
    ax2.set_title("2. Erro Epipolar (Y-Error)")
    ax2.legend()
    
    # Plot 3: Erro por Par
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.bar(np.arange(len(errors_per_pair)), errors_per_pair, color='orange', edgecolor='black')
    ax3.axhline(np.mean(errors_per_pair), color='red', linestyle='dashed', linewidth=2, label="Média")
    ax3.set_title("3. Erro Médio por Imagem")
    ax3.legend()
    
    # Plot 4: 3D Poses Extrinsics
    ax4 = fig.add_subplot(2, 3, (4, 5), projection='3d')
    ax4.scatter(0, 0, 0, c='b', marker='^', s=200, label='Cam Left (Origem)')
    ax4.scatter(T[0], T[1], T[2], c='g', marker='^', s=200, label='Cam Right')
    
    for i in range(len(rvecs_l)):
        R_board, _ = cv2.Rodrigues(rvecs_l[i])
        board_pts = objpoints[i].reshape(-1, 3)
        # Transforma os pontos reais do tabuleiro para a câmera esquerda
        c_trans = (R_board @ board_pts.T).T + tvecs_l[i].T
        ax4.scatter(c_trans[:, 0], c_trans[:, 1], c_trans[:, 2], c='gray', s=5, alpha=0.3)
        # Desenhar bounding box do board
        if len(c_trans) >= 4:
            bnd = [c_trans[0], c_trans[len(c_trans)//2], c_trans[-1]]
            # Apenas pontos representativos para dar contexto
            ax4.plot([b[0] for b in bnd], [b[1] for b in bnd], [b[2] for b in bnd], c='red', alpha=0.5)

    ax4.set_title('4. Poses Espaciais 3D (Câmeras e Tabuleiros)')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_zlabel('Z (mm)')
    ax4.legend()

    # Plot 5: Distorção Radial
    ax5 = fig.add_subplot(2, 3, 6)
    plot_radial_distortion(ax5, mtx_l, dist_l, mtx_r, dist_r, img_size)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"\n[OK] Gráficos de diagnóstico salvos em '{output_file}'")

def main():
    parser = argparse.ArgumentParser(description="Calibrador Estéreo Avançado Metrológico")
    parser.add_argument('--left', type=str, required=True, help="Diretório das imagens da câmera Esquerda")
    parser.add_argument('--right', type=str, required=True, help="Diretório das imagens da câmera Direita")
    parser.add_argument('--cols', type=int, default=11, help="Número de quadrados no eixo X")
    parser.add_argument('--rows', type=int, default=8, help="Número de quadrados no eixo Y")
    parser.add_argument('--square', type=float, default=25.0, help="Tamanho do quadrado em mm")
    parser.add_argument('--marker', type=float, default=18.75, help="Tamanho do marcador interno em mm")
    parser.add_argument('--dict', type=str, default='DICT_5X5_100', help="Dicionário ArUco (ex: DICT_5X5_100)")
    parser.add_argument('--baseline-real', type=float, default=None, help="Valor físico (mm) da Baseline para aferição")
    parser.add_argument('--working-distance', type=float, default=500.0, help="Distância de Trabalho Z (mm) para cálculo da Resolução")
    args = parser.parse_args()

    dict_enum = getattr(cv2.aruco, args.dict, None)
    if dict_enum is None:
        print(f"Erro: Dicionário {args.dict} não encontrado no cv2.aruco.")
        return

    print("Iniciando Calibração Interativa ChArUco...")
    dictionary, board = create_charuco_board(dict_enum, args.cols, args.rows, args.square, args.marker)
    if board is None: return

    left_imgs = sorted(glob.glob(os.path.join(args.left, "*.png")))
    right_imgs = sorted(glob.glob(os.path.join(args.right, "*.png")))

    if len(left_imgs) != len(right_imgs) or len(left_imgs) == 0:
        print("Erro: Quantidade incompatível de imagens entre as pastas L e R.")
        return

    print(f"Encontrados {len(left_imgs)} pares de imagens. Carregando...")
    valid_pairs = []
    img_size = None

    cv2.namedWindow('Calibration (Left | Right)', cv2.WINDOW_NORMAL)
    for i in range(len(left_imgs)):
        img_l = cv2.imread(left_imgs[i])
        img_r = cv2.imread(right_imgs[i])
        gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        if img_size is None: img_size = gray_l.shape[::-1]

        corners_l, ids_l = detect_charuco_corners(gray_l, dictionary, board)
        corners_r, ids_r = detect_charuco_corners(gray_r, dictionary, board)
        com_corners_l, com_corners_r, com_ids = filter_common_corners(corners_l, ids_l, corners_r, ids_r)

        disp_l = img_l.copy()
        disp_r = img_r.copy()
        if com_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(disp_l, com_corners_l, com_ids, (0, 255, 0))
            cv2.aruco.drawDetectedCornersCharuco(disp_r, com_corners_r, com_ids, (0, 255, 0))
            combined = np.hstack((disp_l, disp_r))
            cv2.imshow('Calibration (Left | Right)', combined)
            
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                return
            elif key == ord(' '):
                valid_pairs.append({
                    'id': i, 'corners_l': com_corners_l, 'corners_r': com_corners_r, 'ids': com_ids
                })
                print(f"Par {i} [ACEITO] - {len(com_ids)} pontos em comum.")
            else:
                print(f"Par {i} [DESCARTADO]")
        else:
            print(f"Par {i} [DESCARTADO] - Pontos insuficientes.")
    cv2.destroyAllWindows()

    if len(valid_pairs) < 5:
        print("Erro: Necessário 5 pares válidos.")
        return

    def perform_stereo_calibration(pairs):
        all_corners_l = [p['corners_l'] for p in pairs]
        all_corners_r = [p['corners_r'] for p in pairs]
        all_ids = [p['ids'] for p in pairs]
        
        objpoints = []
        for i in range(len(all_ids)):
            if hasattr(board, 'chessboardCorners'):
                objp = np.array([board.chessboardCorners[idx[0]] for idx in all_ids[i]])
            else:
                objp = np.array([board.getChessboardCorners()[idx[0]] for idx in all_ids[i]])
            objpoints.append(objp)

        ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objpoints, all_corners_l, img_size, None, None)
        ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objpoints, all_corners_r, img_size, None, None)

        mean_errs_l, pt_errs_l = compute_reprojection_errors(objpoints, all_corners_l, rvecs_l, tvecs_l, mtx_l, dist_l)
        mean_errs_r, pt_errs_r = compute_reprojection_errors(objpoints, all_corners_r, rvecs_r, tvecs_r, mtx_r, dist_r)
        
        combined_errs = [(mean_errs_l[i] + mean_errs_r[i])/2 for i in range(len(pairs))]

        flags = cv2.CALIB_FIX_INTRINSIC
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
        
        ret_s, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
            objpoints, all_corners_l, all_corners_r,
            mtx_l, dist_l, mtx_r, dist_r,
            img_size, criteria=criteria, flags=flags)
            
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T)

        y_errors = compute_epipolar_y_error(all_corners_l, all_corners_r, mtx_l, dist_l, mtx_r, dist_r, R1, P1, R2, P2)
        fov_l = calculate_fov_coverage(pt_errs_l, img_size)
        vergence, depth_res = compute_vergence_and_resolution(R, T, mtx_l, args.working_distance)
        
        diagnostics = {
            'y_errors': y_errors,
            'mean_y_error': np.mean(y_errors) if len(y_errors)>0 else 0,
            'max_y_error': np.max(y_errors) if len(y_errors)>0 else 0,
            'pt_errs_l': pt_errs_l,
            'fov_coverage_percent': fov_l,
            'vergence_angle_deg': vergence,
            'theoretical_depth_resolution_mm': depth_res,
            'objpoints': objpoints,
            'rvecs_l': rvecs_l,
            'tvecs_l': tvecs_l
        }

        return ret_s, combined_errs, (mtx_l, dist_l, mtx_r, dist_r, R, T, R1, R2, P1, P2, Q), diagnostics

    final_diag = None
    final_errors = None

    while True:
        rms, errors, params, diag = perform_stereo_calibration(valid_pairs)
        final_diag = diag
        final_errors = errors
        
        print("\n--- DIAGNÓSTICO METROLÓGICO ---")
        print(f"Número de Pares: {len(valid_pairs)}")
        print(f"RMS Estéreo Final: {rms:.4f}")
        print(f"Erro Epipolar Y Médio: {diag['mean_y_error']:.4f} px (Max: {diag['max_y_error']:.4f} px)")
        print(f"Cobertura do Sensor: {diag['fov_coverage_percent']:.1f}%")
        print(f"Ângulo de Vergência: {diag['vergence_angle_deg']:.2f} graus")
        print(f"Resol. Teórica de Profundidade (Z={args.working_distance}mm): {diag['theoretical_depth_resolution_mm']:.4f} mm")
        
        resp = input("\nDigite LIMIAR MÁXIMO (ex: 0.35) para filtrar pares, ou 'S' para Salvar: ").strip()
        if resp.lower() == 's' or resp == '':
            break
        try:
            threshold = float(resp)
            new_pairs = [p for i, p in enumerate(valid_pairs) if errors[i] <= threshold]
            if len(new_pairs) < 5:
                print("Aviso: Limiar deixará menos de 5 pares. Ignorando.")
            else:
                valid_pairs = new_pairs
        except ValueError:
            print("Valor inválido.")

    mtx_l, dist_l, mtx_r, dist_r, R, T, R1, R2, P1, P2, Q = params
    
    # 3D Plots
    generate_diagnostic_plots(
        final_errors, final_diag['y_errors'], final_diag['pt_errs_l'], img_size,
        final_diag['objpoints'], final_diag['rvecs_l'], final_diag['tvecs_l'], R, T,
        mtx_l, dist_l, mtx_r, dist_r
    )

    calc_baseline = np.linalg.norm(T)
    baseline_info = {"calculated_baseline_mm": calc_baseline}
    if args.baseline_real is not None:
        b_error = abs(calc_baseline - args.baseline_real)
        baseline_info["baseline_error_mm"] = b_error
        baseline_info["baseline_error_percent"] = (b_error / args.baseline_real) * 100

    np.savez("stereo_params.npz", mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r, R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
    
    calib_data = {
        "rms_stereo": rms,
        "mean_y_error": final_diag['mean_y_error'],
        "max_y_error": final_diag['max_y_error'],
        "fov_coverage_percent": final_diag['fov_coverage_percent'],
        "vergence_angle_deg": final_diag['vergence_angle_deg'],
        "theoretical_depth_resolution_mm": final_diag['theoretical_depth_resolution_mm'],
        **baseline_info,
        "camera_left": {
            "camera_matrix": mtx_l.tolist(),
            "dist_coeffs": dist_l.tolist(),
            "R1": R1.tolist(),
            "P1": P1.tolist()
        },
        "camera_right": {
            "camera_matrix": mtx_r.tolist(),
            "dist_coeffs": dist_r.tolist(),
            "R1": R2.tolist(),
            "P1": P2.tolist()
        },
        "stereo": {
            "R": R.tolist(),
            "T": T.tolist(),
            "Q": Q.tolist()
        }
    }
    with open("stereo_params.json", "w") as f:
        json.dump(calib_data, f, indent=4)
    print("\n[OK] Parâmetros salvos com sucesso!")

if __name__ == "__main__":
    main()