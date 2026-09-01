import os
import cv2
import numpy as np
import glob
import numpy as np
import open3d as o3d

from msca.processing.stereo_fringe_process import FringeProcess
from msca.processing.InverseTriangulation import InverseTriangulation
from msca.processing.FringePattern import FringePattern
from msca.processing.InverseTriangulation import InverseTriangulation
from msca.config import (
    PROJECTOR_WIDTH, PROJECTOR_HEIGHT, FRINGE_PERIOD_PIXELS, NUM_SHIFTS,
    CAPTURE_DIR, CALIB_FILE,
    X_MIN_COARSE, X_MAX_COARSE, Y_MIN_COARSE, Y_MAX_COARSE, Z_MIN_COARSE, Z_MAX_COARSE, XY_STEP_COARSE, Z_STEP_COARSE, NUM_SPLITS_COARSE,
    MARGIN_XY_REFINED, MARGIN_Z_REFINED, XY_STEP_REFINED, Z_STEP_REFINED, NUM_SPLITS_REFINED
)

def load_images_from_directory(directory):
    """
    Lê os arquivos L*.png e R*.png ordenados do diretório.
    Retorna duas listas de arrays numpy.
    """
    left_files = sorted(glob.glob(os.path.join(directory, 'L*.png')))
    right_files = sorted(glob.glob(os.path.join(directory, 'R*.png')))
    
    if len(left_files) == 0 or len(left_files) != len(right_files):
        raise ValueError(f"Imagens não encontradas ou contagem não coincide. L: {len(left_files)}, R: {len(right_files)}")

    left_imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in left_files]
    right_imgs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in right_files]

    return left_imgs, right_imgs

def main():

    # ----- Parâmetros do Sistema -----
    # ----- Parâmetros do Sistema -----
    capture_dir = CAPTURE_DIR
    calib_file = CALIB_FILE
    
    img_resolution = (PROJECTOR_WIDTH, PROJECTOR_HEIGHT)
    
    # IMPORTANTE: Esta deve ser a resolução DE CAPTURA das Câmeras (e.g. 1920x1200),
    # substitua de acordo com a sua câmera FLIR ou deixe o script ler a partir da primeira imagem.
    
    # ----- Carregar Imagens -----
    print(f"Carregando imagens de {capture_dir}...")
    try:
        left_imgs, right_imgs = load_images_from_directory(capture_dir)
    except Exception as e:
        print(e)
        return
        
    cam_resolution = (left_imgs[0].shape[1], left_imgs[0].shape[0])
    print(f"Resolução da Câmera detectada: {cam_resolution}")

    # ----- Passo 1: Cálculo de Fase (FringeProcess) -----
    print("Iniciando Cálculo da Fase Desembrulhada...")
    stereo = FringeProcess(img_resolution=img_resolution, 
                           camera_resolution=cam_resolution, 
                           px_f=FRINGE_PERIOD_PIXELS, 
                           steps=NUM_SHIFTS)
                           
    for count, (l_img, r_img) in enumerate(zip(left_imgs, right_imgs)):
        # Configurar as imagens na classe de processamento
        stereo.set_images(l_img, r_img, counter=count)

    # Calcular Fase Absoluta (Desembrulhada via GrayCode) e Mapas de Modulação
    abs_phi_image_left, abs_phi_image_right, modulation_mask_left, modulation_mask_right = stereo.calculate_abs_phi_images(visualize=False, save=True)

    # ----- Passo 2: Triangulação Inversa 3D (CuPy) -----
    if not os.path.exists(calib_file):
        print(f"Arquivo de calibração não encontrado: {calib_file}")
        return

    print("Carregando parâmetros de Calibração Estéreo e removendo distorção...")
    zscan = InverseTriangulation(calib_file)
    
    # Remover distorção da lente usando parâmetros estéreo
    abs_phi_image_left_undistorted = zscan.remove_img_distortion(abs_phi_image_left, 'left')
    abs_phi_image_right_undistorted = zscan.remove_img_distortion(abs_phi_image_right, 'right')

    print("Enviando dados de Fase e Máscaras para a GPU...")
    zscan.read_images(left_imgs=abs_phi_image_left_undistorted, 
                      right_imgs=abs_phi_image_right_undistorted, 
                      left_mask=modulation_mask_left, 
                      right_mask=modulation_mask_right)

    # ----- Passo 3: Varredura Espacial (Z-Scan) -----
    # Definir volume de busca em milímetros [min, max, passo]
    x_lin = np.arange(X_MIN_COARSE, X_MAX_COARSE, XY_STEP_COARSE)
    y_lin = np.arange(Y_MIN_COARSE, Y_MAX_COARSE, XY_STEP_COARSE)
    z_lin = np.arange(Z_MIN_COARSE, Z_MAX_COARSE, Z_STEP_COARSE)

    print(f"DEBUG: Modulação Máxima (Esq: {np.max(modulation_mask_left):.2f}, Dir: {np.max(modulation_mask_right):.2f})")
    print(f"DEBUG: Modulação Média (Esq: {np.mean(modulation_mask_left):.2f}, Dir: {np.mean(modulation_mask_right):.2f})")

    # Subdividir para não estourar a memória da GPU
    num_splits = NUM_SPLITS_COARSE
    x_split = np.array_split(x_lin, num_splits)
    y_split = np.array_split(y_lin, num_splits)

    points_result = []
    count = 0
    total_blocos = num_splits * num_splits
    
    print(f"Iniciando Z-Scan grosseiro ({total_blocos} blocos de processamento)...")
    for x_arr in x_split:
        for y_arr in y_split:
            points_3d = zscan.points3D_arrays(x_arr, y_arr, z_lin, visualize=False)
            z_zcan_points = zscan.fringe_process(points_3d=points_3d, save_points=False, visualize=False)
            points_result.append(z_zcan_points)
            count += 1
            print(f"Bloco processado: {count}/{total_blocos}")

    points_result_ar = np.concatenate(points_result, axis=0)
    points_result_ar_filtered = points_result_ar

    # ----- Passo 4: Refinamento Espacial (Z-Scan Fino) -----
    if points_result_ar_filtered.shape[0] == 0:
        print("Atenção: A varredura não encontrou nenhum ponto com validação estéreo. Verifique o posicionamento do alvo ou a calibração.")
        return

    print("Calculando volume delimitador para varredura de refinamento...")
    x_min, x_max = points_result_ar_filtered[:, 0].min() - MARGIN_XY_REFINED, points_result_ar_filtered[:, 0].max() + MARGIN_XY_REFINED
    y_min, y_max = points_result_ar_filtered[:, 1].min() - MARGIN_XY_REFINED, points_result_ar_filtered[:, 1].max() + MARGIN_XY_REFINED
    
    z_min_pre, z_max_pre = points_result_ar_filtered[:, 2].min(), points_result_ar_filtered[:, 2].max()
    z_min = z_min_pre - MARGIN_Z_REFINED
    z_max = z_max_pre + MARGIN_Z_REFINED
    
    print(f"Bounding Box X: {x_min:.1f} to {x_max:.1f}, Y: {y_min:.1f} to {y_max:.1f}, Z: {z_min:.1f} to {z_max:.1f}")

    x_lin_refined = np.arange(x_min, x_max, XY_STEP_REFINED)
    y_lin_refined = np.arange(y_min, y_max, XY_STEP_REFINED)
    z_lin_refined = np.arange(z_min, z_max, Z_STEP_REFINED)
    
    num_splits_refined = NUM_SPLITS_REFINED
    x_split_refined = np.array_split(x_lin_refined, num_splits_refined)
    y_split_refined = np.array_split(y_lin_refined, num_splits_refined)
    
    points_result_refined = []
    count = 0
    print(f"Iniciando Z-Scan Refinado...")
    for x_arr_r in x_split_refined:
        for y_arr_r in y_split_refined:
            points_3d = zscan.points3D_arrays(x_arr_r, y_arr_r, z_lin_refined, visualize=False)
            z_zcan_points = zscan.fringe_process(points_3d=points_3d, save_points=False, visualize=False)
            points_result_refined.append(z_zcan_points)
            count += 1
            print(f"Refinamento bloco: {count}/{total_blocos}")

    if len(points_result_refined) > 0:
        points_result_refined_ar = np.vstack(points_result_refined)
        print("Z values BEFORE filtering:")
        print("Min Z:", points_result_refined_ar[:, 2].min())
        print("Max Z:", points_result_refined_ar[:, 2].max())
        np.savetxt('unfiltered_points.txt', points_result_refined_ar, fmt='%.8f')
    else:
        points_result_refined_ar = np.array([])
    
    print("Filtrando e removendo outliers da malha de pontos usando Open3D...")
    points_result_refined_ar_filtered = zscan.filter_points_by_depth(points_result_refined_ar, depth_threshold=2.0)
    final_point_cloud = points_result_refined_ar_filtered

    # Salvar resultados
    output_filename = 'fringe_points_cloud.txt'
    np.savetxt(output_filename, final_point_cloud, fmt='%.8f', delimiter=' ')
    print(f"Processamento completo. Nuvem de pontos salva em: {output_filename} ({final_point_cloud.shape[0]} pontos gerados).")

    # Exibir Nuvem
    zscan.plot_3d_points(final_point_cloud[:, 0], final_point_cloud[:, 1], final_point_cloud[:, 2], title='Nuvem de Pontos Final (Fringe Projection)')

if __name__ == '__main__':
    main()
