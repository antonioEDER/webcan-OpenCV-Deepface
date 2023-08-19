# Reconhecimento de Emoções e Carros em Tempo Real usando OpenCV e Deepface

Este projeto demonstra a implementação de reconhecimento de emoções faciais e carros em tempo real usando a biblioteca deepface e o OpenCV. O objetivo é capturar vídeo ao vivo de uma webcam, identificar rostos no fluxo de vídeo e prever as emoções correspondentes para cada rosto detectado. As emoções previstas são exibidas em tempo real nos quadros de vídeo.

Para simplificar esse processo, utilizamos a biblioteca deepface, uma ferramenta de análise facial baseada em aprendizado profundo que emprega modelos pré-treinados para detecção precisa de emoções. Além disso, aproveitamos o OpenCV, uma biblioteca de visão computacional de código aberto, para facilitar o processamento de imagens e vídeos.

## Instruções

### Configuração Inicial::

1. Clone o repositório: Execute `git clone https://github.com/antonioEDER/webcan-OpenCV-Deepface.git`.

2. Acesse o diretório do projeto: Execute `cd webcan-OpenCV-Deepface`.

3. Instale as dependências necessárias:
   -  1: Use `pip install -r requirements.txt`.
   -  2: Instale as dependências individualmente:
     - `pip install deepface`
     - `pip install opencv-python`

4. Obtenha o arquivo XML da cascata Haar para detecção de rostos::
   - Baixe o arquivo `haarcascade_frontalface_default.xml` disponível no [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

5. Obtenha o arquivo XML da cascata Haar para detecção de carros::
   - Baixe o arquivo `cars.xml` disponível no [andrewssobral GitHub repository](https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml)  e renomear para vehicle.xml.

6. Execute o código:
   - Execute o script em Python.
   - A webcam será ativada, iniciando a detecção em tempo real das emoções faciais.
   - As etiquetas de emoção serão sobrepostas nos quadros que contêm rostos reconhecidos.

## Comandos
1. Abrir Webcan e ativar reconhecimento de emoções faciais:
   -  1: Use `sudo python emotion.py`.
2. Abrir Webcan e ativar reconhecimento de carros:
   -  1: Use `sudo python emotion.py`.

## Abordagem

1. Importar Bibliotecas Essenciais: Importar cv2 para captura de vídeo e processamento de imagens, além de deepface para o modelo de detecção de emoções.

2. Carregar o Classificador de Cascata Haar: Utilizar cv2.CascadeClassifier() para carregar o arquivo XML de detecção de rostos/carros.

3. Inicialização da Captura de Vídeo: Utilizar cv2.VideoCapture() para iniciar a captura de vídeo da webcam padrão.

4. Loop de Processamento de Quadros: Entrar em um loop contínuo para processar cada quadro de vídeo.

5. Conversão em Tons de Cinza: Transformar cada quadro em tons de cinza usando cv2.cvtColor().

6. Detecção de Rostos: Detectar rostos no quadro em tons de cinza usando face_cascade.detectMultiScale().

7. Extração da Região do Rosto: Para cada rosto detectado, extrair a Região de Interesse (ROI) contendo o rosto.

8. Pré-processamento: Preparar a imagem do rosto para detecção de emoções utilizando a função de pré-processamento da biblioteca deepface.

9. Previsão de Emoções: Utilizar o modelo de detecção de emoções pré-treinado fornecido pela biblioteca deepface para prever emoções.

10. Rotulagem de Emoções: Mapear o índice de emoção previsto para a etiqueta de emoção correspondente.

11. Anotação Visual: Desenhar retângulos ao redor dos rostos detectados e rotulá-los com as emoções previstas usando cv2.rectangle() e cv2.putText().

12. Exibir Saída: Apresentar o quadro resultante com a emoção rotulada usando cv2.imshow().

13. Limpeza: Liberar recursos de captura de vídeo e fechar todas as janelas com cap.release() e cv2.destroyAllWindows().
