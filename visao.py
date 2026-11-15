# Detector de luminárias do LuzViva: localiza cada luminária na imagem e indica se está acesa ou apagada
# contar() resume o dataset, treinar() treina o YOLO, detectar() e desenhar() usam o modelo (notebook e app.py)
#
# Dataset (pasta dataset/, formato YOLO, dividido em train/val/test):
# - Lamp State Detection (Hugging Face, CC BY 4.0): 8 dos 10 vídeos (2 têm neve), 1 a cada 6 quadros
#   (1 a cada 2 com luminária apagada), divisão 70/15/15 em blocos de 10 quadros. Arquivos: <vídeo>_<quadro>.jpg
# - 100 imagens do Roboflow (CC BY 4.0), só no treino, para cobrir cenas noturnas: noite_XX (70, street_light_on),
#   apagada_XX (26, street-light-off, anotadas pela equipe) e dia_XX (4, street-light-off)

import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

PASTA = Path(__file__).parent
DATASET = PASTA / "dataset"
CLASSES = ["acesa", "apagada"]
CORES = {"acesa": (40, 170, 40), "apagada": (40, 40, 220)}  # verde e vermelho (o OpenCV usa BGR)
CINZA = (150, 150, 150)
LIMIAR = 0.4  # confiança mínima para o modelo "ter certeza"; abaixo disso a caixa fica cinza


# Conta as imagens e as caixas de cada classe em treino, validação e teste
# (cada imagem tem um .txt em labels/ com uma linha por luminária: classe x_centro y_centro largura altura)
def contar():
    contagem = {}
    for parte in ("train", "val", "test"):
        linhas = [l for txt in (DATASET / parte / "labels").glob("*.txt") for l in txt.read_text().splitlines() if l.strip()]
        contagem[parte] = {"imagens": len(list((DATASET / parte / "images").glob("*.jpg"))),
                           "acesa": sum(l.split()[0] == "0" for l in linhas),
                           "apagada": sum(l.split()[0] == "1" for l in linhas)}
    return contagem


# Treina o YOLO11n (pré-treinado no COCO) para as 2 classes do projeto
# e guarda o melhor modelo em modelo/luzviva.pt e o histórico em resultados/treino.csv
def treinar(epocas=100):
    yolo = YOLO(PASTA / "yolo11n.pt")
    yolo.train(data=DATASET / "data.yaml", epochs=epocas, imgsz=640, batch=8, patience=20, seed=42,
               workers=4, project=PASTA / "runs", name="treino", exist_ok=True)
    (PASTA / "modelo").mkdir(exist_ok=True)
    (PASTA / "resultados").mkdir(exist_ok=True)
    shutil.copy(PASTA / "runs" / "treino" / "weights" / "best.pt", PASTA / "modelo" / "luzviva.pt")
    shutil.copy(PASTA / "runs" / "treino" / "results.csv", PASTA / "resultados" / "treino.csv")


modelo = None


# Acha as luminárias na imagem e devolve a classe, a confiança (0 a 1) e a caixa (x1, y1, x2, y2) de cada uma
# (abaixo de 0,2 é só ruído; entre 0,2 e LIMIAR a caixa vem, mas é desenhada em cinza como "incerta")
def detectar(foto, conf=0.2):
    global modelo
    if modelo is None:  # carrega o modelo só na primeira vez
        modelo = YOLO(PASTA / "modelo" / "luzviva.pt")
    r = modelo.predict(foto, conf=conf, verbose=False)[0]
    return [{"classe": r.names[int(c)], "confianca": round(float(p), 2), "caixa": [int(v) for v in b]}
            for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())]


# Desenha as caixas na imagem: verde para acesa, vermelho para apagada e cinza quando a confiança é baixa
def desenhar(foto, luminarias):
    foto = foto.copy()
    for l in luminarias:
        x1, y1, x2, y2 = l["caixa"]
        cor = CORES[l["classe"]] if l.get("confianca", 1) >= LIMIAR else CINZA
        texto = l["classe"] + (f" {l['confianca']:.2f}" if "confianca" in l else "")  # marcação do dataset não tem confiança
        (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(foto, (x1, y1), (x2, y2), cor, 2)
        cv2.rectangle(foto, (x1, y1 - h - 8), (x1 + w + 6, y1), cor, -1)
        cv2.putText(foto, texto, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return foto
