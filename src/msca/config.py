# ==============================================================================
# Configurações Globais do Algoritmo de Projeção de Franjas (FPP)
# ==============================================================================

# ----- Parâmetros do Projetor e Padrão Estruturado -----
# PROJECTOR_WIDTH / PROJECTOR_HEIGHT: A resolução física (em pixels) do projetor utilizado.
PROJECTOR_WIDTH = 1280
PROJECTOR_HEIGHT = 720
# FRINGE_PERIOD_PIXELS: A largura (em pixels) de cada "onda" senoidal projetada na tela.
FRINGE_PERIOD_PIXELS = 20  
# NUM_SHIFTS: O número de passos de fase utilizados (N-step phase shift).
NUM_SHIFTS = 12            

# ----- Parâmetros de Entrada e Calibração -----
# CAPTURE_DIR: O diretório onde as imagens (L*.png e R*.png) lidas pelo processamento offline estão salvas.
CAPTURE_DIR = "out/capture_fpp"
# CALIB_FILE: O arquivo (.npz) contendo a calibração intrínseca, extrínseca e as matrizes de distorção.
CALIB_FILE = "stereo_params.npz"

# ----- Parâmetros de Varredura Grosseira (Coarse Z-Scan) -----
# O Coarse Z-Scan varre um volume grande no espaço para encontrar a peça bruta.
# Limites X (Largura) e Y (Altura) do volume de busca inicial (em mm)
X_MIN_COARSE = -100.0
X_MAX_COARSE = 100.0
Y_MIN_COARSE = -100.0
Y_MAX_COARSE = 200.0
# Limites Z (Profundidade) do volume de busca (em mm). Deve envolver a peça (ex: 1.5m = 1500mm).
Z_MIN_COARSE = 1100.0
Z_MAX_COARSE = 1220.0
# Passo (step) entre os planos da varredura. Um passo maior poupa memória/processamento na busca inicial.
XY_STEP_COARSE = 5.0
Z_STEP_COARSE = 10.0
# NUM_SPLITS_COARSE: Divide a varredura grosseira em sub-blocos (10x10=100 blocos) para evitar estouro de memória RAM.
NUM_SPLITS_COARSE = 10     

# ----- Parâmetros de Varredura Fina (Refined Z-Scan) -----
# O Z-Scan Refinado pega a 'bounding box' encontrada pela busca grosseira e faz uma varredura minuciosa.
# Margens de segurança expandidas ao redor da bounding box (em mm) para garantir que a peça inteira será lida.
MARGIN_XY_REFINED = 2.0    
MARGIN_Z_REFINED = 5.0     
# Passo ultrafino (resolução) da nuvem de pontos final (em mm). Z_STEP define a "resolução de profundidade".
XY_STEP_REFINED = 1.0
Z_STEP_REFINED = 0.1
# Divisão em sub-blocos da etapa refinada.
NUM_SPLITS_REFINED = 10

# ----- Filtros e Limiares (Máscara e Interpolação) -----
# MODULATION_THRESHOLD_FACTOR: Fator limitante do ruído. Se o contraste/modulação da franja for menor que
# (este fator * a modulação média), o ponto é ignorado. Ajuda a filtrar fundos pretos e sombras.
MODULATION_THRESHOLD_FACTOR = 0.5  
# Limiares de Desvio Padrão da Interpolação: Filtram "artefatos" (ex: bordas irregulares e ruídos subpixel). 
# Valores baixos (ex: 1.0) garantem que só pontos com interpolação bicúbica de alta confiança sejam mantidos na nuvem.
STD_MIN_THRESH = 0.0               
STD_MAX_THRESH = 1.0               

# ----- Filtros Pós-processamento da Nuvem de Pontos e RANSAC (Open3D) -----
# STATISTICAL_OUTLIER: Limpa ruídos soltos ou "fantasmas" que flutuam em volta da nuvem principal.
# Requer um número mínimo de vizinhos num certo raio de desvio (std_ratio).
STATISTICAL_OUTLIER_NEIGHBORS = 300
STATISTICAL_OUTLIER_STD_RATIO = 0.1

# RANSAC: Algoritmo usado para identificar, de forma automatizada, qual é o "plano do fundo" (ex: a parede).
# Ao achar a parede, a nuvem de pontos é rotacionada e posta reta (Z=0).
RANSAC_DISTANCE_THRESHOLD = 5.0    # Tolerância máxima de erro para um ponto pertencer ao plano base (em mm).
RANSAC_N = 3                       # Pontos mínimos para estimar um plano.
RANSAC_NUM_ITERATIONS = 1000       # Quantas vezes o algoritmo vai tentar chutar um plano antes de desistir.
