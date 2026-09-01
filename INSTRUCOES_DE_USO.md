# Guia de Utilização - Processamento Estéreo FPP (Offline)

Este documento fornece as instruções passo a passo para executar o algoritmo de metrologia óptica FPP (Fringe Projection Profilometry) 3D puramente por software. 

A captura de imagens foi desacoplada deste projeto. Este repositório foca 100% no motor matemático (reconstrução da Nuvem de Pontos através de processamento massivo via CPU/Numpy).

---

## 1. Visão Geral da Arquitetura

O sistema funciona importando as imagens previamente capturadas de um alvo estéreo (por qualquer hardware que você desejar). O algoritmo calcula o mapa de fase, mapeia a intersecção estéreo (Triangulação Inversa / Z-Scan) e filtra as informações para gerar um modelo 3D altamente denso.

---

## 2. Configuração do Ambiente

**Dependências Requeridas:**
* `numpy`
* `opencv-python`
* `opencv-contrib-python` (Necessário se for calibrar os parâmetros com matrizes Aruco)
* `matplotlib`
* `open3d` (Para manipulação final e visualização da nuvem de pontos)

*(Nota: Ferramentas pesadas baseadas em CUDA ou bibliotecas de hardware/sensores foram completamente eliminadas, garantindo portabilidade para qualquer máquina, incluindo Macs Apple Silicon).*

---

## 3. Passo a Passo de Execução

### Passo 1: Preparação das Imagens e Calibração
Antes de reconstruir, você precisa fornecer as entradas matemáticas:
1. **Imagens Capturadas**: Coloque as imagens das franjas das câmeras esquerda (`L*.png`) e direita (`R*.png`) na pasta apontada pela variável `CAPTURE_DIR` (o padrão é `out/capture_fpp`).
2. **Matrizes de Calibração**: O algoritmo necessita da calibração estéreo (intrínsecos e extrínsecos) gerada por um calibrador. Salve o arquivo `.npz` (o padrão é `stereo_params.npz`) na raiz do projeto.

### Passo 2: Configuração dos Padrões (FPP)
Diferente das versões anteriores, **NÃO** edite os scripts matemáticos. Todas as constantes do sistema foram centralizadas.
1. Abra o arquivo `src/msca/config.py`.
2. Lá você encontrará todas as configurações divididas por categorias:
   - **Projetor:** Resolução, passos de fase (N-step) e período (largura) da franja.
   - **Varredura (Z-Scan):** Você **deve** garantir que os limites da varredura (`Z_MIN_COARSE` a `Z_MAX_COARSE`) englobam a distância física real onde o seu alvo estava posicionado durante a captura.
   - **Filtros e Limiares:** Afrouxe ou aperte a tolerância ao ruído alterando o fator de modulação e o desvio padrão de interpolação aceitos.

### Passo 3: Reconstrução 3D
Execute o orquestrador matemático para computar as franjas e gerar o modelo 3D.
1. **No seu terminal, ative o ambiente virtual e execute o script:**
   ```bash
   # Ative o ambiente virtual
   source tmp_venv/bin/activate
   
   # Execute o código mestre
   PYTHONPATH=src python3 main_reconstruction.py
   ```
2. **Resultado:** 
   O sistema computará as matrizes, realizará uma varredura grosseira do espaço, depois uma varredura refinada baseada na bounding box gerada, e removerá *outliers*. Ao final, irá exibir um gráfico 3D interativo da nuvem de pontos, além de exportá-la no arquivo `fringe_points_cloud.txt` para você poder visualizar em softwares como Meshlab ou CloudCompare.

---

## Dicas Adicionais e Solução de Problemas

* **Erro de Falta de Pontos (`Dentro da Câmera (FOV): Esq: 0, Dir: 0`):** Isso indica que o algoritmo Z-Scan está varrendo um volume 3D mas o cruzamento da câmera esquerda com a direita está fora de quadro (as câmeras não estão convergindo no ponto). Revise as matrizes de Translação e Rotação geradas na sua calibração (`stereo_params.npz`), elas costumam ser as grandes vilãs. Outra causa pode ser os limites de busca `Z_MIN` e `Z_MAX` no `config.py` não estarem cobrindo o objeto (ex: a varredura está de 0 a 50mm mas o alvo está a 1500mm).
* **Falta de Memória RAM na Reconstrução:** A matriz de pesquisa do algoritmo Z-Scan exige muita RAM. O código subdivide a varredura em pequenos blocos (através das variáveis `NUM_SPLITS_COARSE` e `NUM_SPLITS_REFINED` do `config.py`). Se o seu PC matar o processo (`Killed: 9`), experimente dobrar os splits para dividir a carga na memória.
