"""
Chat de grupo anónimo.

Cualquiera puede entrar a una sala con solo abrir el enlace (o con un código de
sala). A cada persona se le asigna un alias anónimo aleatorio y un color; nadie
en el chat sabe quién es quién: no hay registro, ni nombres reales, ni emails,
ni se muestran direcciones IP.

Ejecutar:
    python chat_app.py
Luego abrir http://localhost:5000 en el navegador. Comparte esa URL (o el
código de sala) con la gente que quieras meter al chat.
"""

import os
import secrets
import threading
import time
from collections import defaultdict, deque

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

app = Flask(__name__)
# Clave de sesión: usada solo para firmar la cookie que guarda el alias anónimo.
app.secret_key = os.environ.get("CHAT_SECRET_KEY", secrets.token_hex(32))

# --- Generación de alias anónimos --------------------------------------------
ADJETIVOS = [
    "Silencioso", "Curioso", "Veloz", "Sereno", "Oculto", "Errante", "Astuto",
    "Brillante", "Nocturno", "Valiente", "Misterioso", "Tranquilo", "Ingenioso",
    "Salvaje", "Cordial", "Fugaz", "Sabio", "Audaz", "Sutil", "Libre",
]
ANIMALES = [
    "Zorro", "Búho", "Lobo", "Halcón", "Panda", "Lince", "Cuervo", "Tigre",
    "Delfín", "Erizo", "Gato", "Nutria", "Tejón", "Ciervo", "Colibrí",
    "Mapache", "Pingüino", "Camaleón", "Mantis", "Koala",
]
COLORES = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#800000", "#808000", "#000075", "#e6beff", "#aaffc3", "#ffd8b1",
]


def nuevo_alias():
    """Devuelve un (nombre, color) anónimo aleatorio."""
    nombre = "{} {} #{:03d}".format(
        secrets.choice(ADJETIVOS),
        secrets.choice(ANIMALES),
        secrets.randbelow(1000),
    )
    color = secrets.choice(COLORES)
    return nombre, color


# --- Almacén de mensajes en memoria ------------------------------------------
# Estructura: rooms[codigo_sala] = deque de mensajes.
# No se persiste nada en disco: al reiniciar el servidor, el historial se borra.
_lock = threading.Lock()
rooms = defaultdict(lambda: deque(maxlen=500))
_next_id = defaultdict(int)  # id incremental por sala, para el polling


def sala_actual():
    """Código de sala pedido (?room=xxx) o 'general' por defecto."""
    code = (request.args.get("room") or request.form.get("room") or "general").strip()
    # Solo caracteres seguros; evitamos que el código rompa nada.
    code = "".join(c for c in code if c.isalnum() or c in "-_")[:32]
    return code or "general"


def asegurar_identidad():
    """Garantiza que la sesión tenga un alias anónimo asignado."""
    if "alias" not in session or "color" not in session:
        nombre, color = nuevo_alias()
        session["alias"] = nombre
        session["color"] = color
    # Un id de participante estable pero anónimo (no revela nada real).
    if "pid" not in session:
        session["pid"] = secrets.token_hex(8)


@app.route("/")
def index():
    asegurar_identidad()
    room = sala_actual()
    return render_template(
        "chat.html",
        room=room,
        alias=session["alias"],
        color=session["color"],
    )


@app.route("/messages")
def messages():
    """Devuelve los mensajes nuevos desde ?since=<id>. Usado por el polling."""
    room = sala_actual()
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        since = 0
    with _lock:
        nuevos = [m for m in rooms[room] if m["id"] > since]
    return jsonify(messages=nuevos)


@app.route("/send", methods=["POST"])
def send():
    asegurar_identidad()
    room = sala_actual()
    texto = (request.form.get("text") or "").strip()
    if not texto:
        return jsonify(ok=False, error="mensaje vacío"), 400
    if len(texto) > 2000:
        texto = texto[:2000]

    with _lock:
        _next_id[room] += 1
        msg = {
            "id": _next_id[room],
            "alias": session["alias"],
            "color": session["color"],
            "pid": session["pid"],  # solo para que el navegador marque "tú"
            "text": texto,
            "ts": time.strftime("%H:%M"),
        }
        rooms[room].append(msg)
    return jsonify(ok=True, message=msg)


@app.route("/nueva-identidad", methods=["POST"])
def nueva_identidad():
    """Regenera el alias anónimo (por si alguien quiere 'reaparecer' de cero)."""
    nombre, color = nuevo_alias()
    session["alias"] = nombre
    session["color"] = color
    session["pid"] = secrets.token_hex(8)
    return jsonify(ok=True, alias=nombre, color=color)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
