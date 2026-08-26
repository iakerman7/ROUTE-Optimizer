"""
Chat de grupo anónimo.

Cualquiera puede entrar a una sala con solo abrir el enlace (o con un código de
sala). A cada persona se le asigna un alias anónimo aleatorio (animal + adjetivo
+ emoji de avatar) y un color; nadie en el chat sabe quién es quién: no hay
registro, ni nombres reales, ni emails, ni se muestran direcciones IP.

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

from flask import Flask, jsonify, render_template, request, session

app = Flask(__name__)
# Clave de sesión: usada solo para firmar la cookie que guarda el alias anónimo.
app.secret_key = os.environ.get("CHAT_SECRET_KEY", secrets.token_hex(32))

# --- Generación de alias anónimos --------------------------------------------
# Adjetivos en forma masculina o invariable: concordar() los pasa a femenino
# cuando el animal lo pide (Ballena Silenciosa, no Ballena Silencioso).
ADJETIVOS = [
    "Silencioso", "Curioso", "Veloz", "Sereno", "Oculto", "Errante", "Astuto",
    "Brillante", "Nocturno", "Valiente", "Misterioso", "Tranquilo", "Ingenioso",
    "Salvaje", "Cordial", "Fugaz", "Sabio", "Audaz", "Sutil", "Libre",
    "Elegante", "Risueño", "Travieso", "Solitario", "Discreto", "Furtivo",
    "Pensativo", "Sigiloso", "Amable", "Distante",
]

# (nombre, emoji del avatar, género gramatical)
ANIMALES = [
    ("Zorro", "🦊", "m"), ("Búho", "🦉", "m"), ("Lobo", "🐺", "m"),
    ("Halcón", "🦅", "m"), ("Panda", "🐼", "m"), ("Lince", "🐆", "m"),
    ("Cuervo", "🐦", "m"), ("Tigre", "🐯", "m"), ("Delfín", "🐬", "m"),
    ("Erizo", "🦔", "m"), ("Gato", "🐱", "m"), ("Tejón", "🦡", "m"),
    ("Ciervo", "🦌", "m"), ("Mapache", "🦝", "m"), ("Pingüino", "🐧", "m"),
    ("Camaleón", "🦎", "m"), ("Koala", "🐨", "m"), ("Pulpo", "🐙", "m"),
    ("Cangrejo", "🦀", "m"), ("Caracol", "🐌", "m"), ("Loro", "🦜", "m"),
    ("Flamenco", "🦩", "m"), ("Oso", "🐻", "m"), ("Elefante", "🐘", "m"),
    ("Búfalo", "🦬", "m"), ("Murciélago", "🦇", "m"), ("Dragón", "🐲", "m"),
    ("Unicornio", "🦄", "m"), ("Pavo", "🦃", "m"), ("Mono", "🐵", "m"),
    ("Nutria", "🦦", "f"), ("Ballena", "🐳", "f"), ("Abeja", "🐝", "f"),
    ("Mariposa", "🦋", "f"), ("Rana", "🐸", "f"), ("Ardilla", "🐿️", "f"),
    ("Llama", "🦙", "f"), ("Foca", "🦭", "f"), ("Cebra", "🦓", "f"),
    ("Tortuga", "🐢", "f"), ("Jirafa", "🦒", "f"), ("Serpiente", "🐍", "f"),
    ("Oveja", "🐑", "f"), ("Mantis", "🦗", "f"),
]

# Colores para el círculo del avatar. Elegidos para que el emoji se lea bien
# tanto en modo claro como oscuro.
COLORES = [
    "#FF6B6B", "#F7913A", "#E8B62C", "#5BC236", "#1FBFA0", "#3AA0FF",
    "#5B6BF5", "#9B5DE5", "#E465C0", "#F2637A", "#00A8A8", "#A87E4F",
    "#7C8DF5", "#43B77A", "#D9784E", "#B76FD9",
]


def concordar(adjetivo, genero):
    """Pasa el adjetivo a femenino si hace falta (los invariables no cambian)."""
    if genero == "f" and adjetivo.endswith("o"):
        return adjetivo[:-1] + "a"
    return adjetivo


# --- Almacén en memoria -------------------------------------------------------
# Estructura: rooms[codigo_sala] = deque de mensajes.
# No se persiste nada en disco: al reiniciar el servidor, el historial se borra.
_lock = threading.Lock()
rooms = defaultdict(lambda: deque(maxlen=500))
_next_id = defaultdict(int)  # id incremental por sala, para el polling
_alias_en_uso = set()  # para no repetir el mismo alias entre participantes


def nuevo_alias():
    """Devuelve un (nombre, emoji, color) anónimo que nadie más esté usando."""
    for _ in range(60):
        animal, emoji, genero = secrets.choice(ANIMALES)
        nombre = "{} {}".format(animal, concordar(secrets.choice(ADJETIVOS), genero))
        with _lock:
            if nombre in _alias_en_uso:
                continue
            # No hay forma de saber cuándo alguien se va (no hay presencia), así
            # que vaciamos el registro si crece demasiado en vez de agotarlo.
            if len(_alias_en_uso) > 600:
                _alias_en_uso.clear()
            _alias_en_uso.add(nombre)
        return nombre, emoji, secrets.choice(COLORES)

    # Combinaciones agotadas: desempatamos con un número.
    animal, emoji, genero = secrets.choice(ANIMALES)
    nombre = "{} {} {}".format(
        animal,
        concordar(secrets.choice(ADJETIVOS), genero),
        secrets.randbelow(90) + 10,
    )
    with _lock:
        _alias_en_uso.add(nombre)
    return nombre, emoji, secrets.choice(COLORES)


def liberar_alias(nombre):
    """Devuelve un alias al pozo (al regenerar identidad)."""
    with _lock:
        _alias_en_uso.discard(nombre)


def sala_actual():
    """Código de sala pedido (?room=xxx) o 'general' por defecto."""
    code = (request.args.get("room") or request.form.get("room") or "general").strip()
    # Solo caracteres seguros; evitamos que el código rompa nada.
    code = "".join(c for c in code if c.isalnum() or c in "-_")[:32]
    return code or "general"


def asegurar_identidad():
    """Garantiza que la sesión tenga un alias anónimo asignado."""
    if not all(k in session for k in ("alias", "color", "emoji")):
        nombre, emoji, color = nuevo_alias()
        session["alias"] = nombre
        session["emoji"] = emoji
        session["color"] = color
    # Un id de participante estable pero anónimo (no revela nada real). Sirve
    # para que el navegador sepa cuáles burbujas son propias.
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
        emoji=session["emoji"],
        color=session["color"],
        pid=session["pid"],
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
            "emoji": session["emoji"],
            "color": session["color"],
            "pid": session["pid"],  # solo para que el navegador marque "tú"
            "text": texto,
            # Epoch en ms: el navegador lo formatea en la zona horaria de cada
            # quien (el servidor de Render corre en UTC).
            "t": int(time.time() * 1000),
        }
        rooms[room].append(msg)
    return jsonify(ok=True, message=msg)


@app.route("/nueva-identidad", methods=["POST"])
def nueva_identidad():
    """Regenera el alias anónimo (por si alguien quiere 'reaparecer' de cero)."""
    if "alias" in session:
        liberar_alias(session["alias"])
    nombre, emoji, color = nuevo_alias()
    session["alias"] = nombre
    session["emoji"] = emoji
    session["color"] = color
    session["pid"] = secrets.token_hex(8)
    return jsonify(ok=True, alias=nombre, emoji=emoji, color=color, pid=session["pid"])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
