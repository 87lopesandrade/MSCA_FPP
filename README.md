
 **[Clique aqui para acessar o Guia de Utilização (INSTRUCOES_DE_USO.md)](INSTRUCOES_DE_USO.md)**

## Dependências

# MSCA FPP - Metrologia Óptica Estéreo (Processamento Offline)

Este repositório contém um sistema puramente matemático (software) de metrologia óptica 3D baseado em Projeção de Franjas (Fringe Projection Profilometry - FPP).

O projeto foi inteiramente refatorado para alta estabilidade, precisão e compatibilidade. Ele processa offline imagens pré-capturadas e exporta nuvens de pontos densas, utilizando operações vetorizadas via NumPy (rodando de forma universal e estável em processadores convencionais, sem dependência de placas de vídeo NVIDIA/CUDA).

## Principais Recursos

- **Processamento Estéreo**: Lê conjuntos de imagens sincronizadas das câmeras esquerda e direita e calcula o mapa de fase 2D absoluto e máscaras de modulação.
- **Triangulação Inversa (Z-Scan)**: Transforma os mapas de fase diretamente em uma densa nuvem de pontos 3D baseada em calibração estéreo (matrizes Intrínsecas e Extrínsecas).
- **Voxelização e RANSAC Integrados**: O algoritmo identifica automaticamente o plano de fundo da captura, alinha a peça e filtra particulados e pontos ruidosos isolados via Open3D.
- **Configuração Centralizada**: Todos os parâmetros matemáticos, limiares de tolerância, filtros estatísticos e partições de memória estão organizados e documentados em um único arquivo de fácil acesso.

## Estrutura do Projeto

O motor matemático está concentrado no pacote nativo `src/msca/`:

- `src/msca/config.py`: **O coração do projeto**. É aqui que você define as dimensões do projetor, os limites do espaço 3D de varredura (Z-Scan) e afina a tolerância do ruído estatístico.
- `src/msca/processing/`: As rotinas pesadas (FringePattern, GrayCode, InverseTriangulation).
- `main_reconstruction.py`: O orquestrador principal. Ele consome as imagens brutas em `out/capture_fpp` e exporta a nuvem formatada em `fringe_points_cloud.txt`.

## Como Usar?

Para rodar sua primeira reconstrução, consulte o guia passo a passo completo:
 **[Clique aqui para acessar o Guia de Utilização (INSTRUCOES_DE_USO.md)](INSTRUCOES_DE_USO.md)**

## Dependências

O sistema é robusto porém enxuto. Instale usando o `requirements.txt` ou ativando o ambiente virtual padrão. O stack principal consiste em:

- `numpy`
- `opencv-python`
- `matplotlib`
- `open3d`
