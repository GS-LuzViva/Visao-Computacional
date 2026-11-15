# Site do detector: analisa uma foto (de exemplo ou enviada) como se fosse um quadro da câmera
# Rodar:  uvicorn app:app --port 8002   e abrir http://localhost:8002

import base64
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, Response

from visao import PASTA, desenhar, detectar

app = FastAPI(title="LuzViva Visão")

# Fotos de exemplo (5 do teste, que o modelo nunca viu, e 1 noturna do treino) e a área que o site mostra (x, y, largura, altura).
# O modelo analisa o quadro inteiro, como a câmera na rua, e o site mostra só a área das luminárias (todas em 16:9)
RECORTES = {
    "01_apagada_6_00390.jpg": (24, 4, 596, 335),
    "02_apagada_7_00348.jpg": (250, 7, 325, 183),
    "03_acesa_10_00363.jpg": (357, 10, 922, 519),
    "04_acesa_10_00340.jpg": (4, 0, 882, 496),
    "05_acesa_9_00228.jpg": (8, 0, 927, 521),
    "06_apagada_noite_14.jpg": (60, 40, 452, 254),  # noturna do Roboflow anotada pela equipe (usada no treino)
}


# Carrega o modelo em segundo plano assim que o site sobe (senão o primeiro clique demora uns 30 s)
@app.on_event("startup")
def aquecer():
    threading.Thread(target=detectar, args=(np.zeros((640, 640, 3), np.uint8),)).start()


# Lê uma foto de exemplo (np.fromfile porque o cv2.imread não abre caminhos com acento, como "Visão")
def abrir_exemplo(nome):
    return cv2.imdecode(np.fromfile(PASTA / "exemplos" / nome, np.uint8), cv2.IMREAD_COLOR)


# Roda o modelo, mede o tempo e monta a resposta com as luminárias e a foto desenhada
# (com recorte, só ficam as luminárias dentro da área mostrada)
def analisar(imagem, recorte=None):
    inicio = time.time()
    luminarias = detectar(imagem)
    tempo = round((time.time() - inicio) * 1000)
    desenhada = desenhar(imagem, luminarias)
    if recorte:
        x, y, w, h = recorte
        luminarias = [l for l in luminarias
                      if x <= (l["caixa"][0] + l["caixa"][2]) / 2 <= x + w and y <= (l["caixa"][1] + l["caixa"][3]) / 2 <= y + h]
        desenhada = desenhada[y:y + h, x:x + w]
    _, jpg = cv2.imencode(".jpg", desenhada)
    return {"luminarias": luminarias, "tempo_ms": tempo,
            "imagem": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()}


# Abre a página do site
@app.get("/")
def site():
    return FileResponse(PASTA / "index.html")


# Lista as fotos de exemplo
@app.get("/lista_exemplos")
def lista_exemplos():
    return list(RECORTES)


# Miniatura de um exemplo: só a área recortada
@app.get("/exemplos/{nome}")
def miniatura(nome: str):
    x, y, w, h = RECORTES[nome]
    _, jpg = cv2.imencode(".jpg", abrir_exemplo(nome)[y:y + h, x:x + w])
    return Response(jpg.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# Analisa um exemplo: o modelo vê o quadro inteiro e o resultado volta recortado
@app.post("/detectar_exemplo/{nome}")
def rota_detectar_exemplo(nome: str):
    return analisar(abrir_exemplo(nome), RECORTES[nome])


# Recebe uma foto enviada, roda o modelo e devolve as luminárias e a foto com as caixas (a foto não é salva)
@app.post("/detectar")
async def rota_detectar(foto: UploadFile):
    return analisar(cv2.imdecode(np.frombuffer(await foto.read(), np.uint8), cv2.IMREAD_COLOR))
