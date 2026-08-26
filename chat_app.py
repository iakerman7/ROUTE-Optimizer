"""
Chat de grupo anónimo.

Cualquiera puede entrar a una sala con solo abrir el enlace (o con un código de
sala). A cada persona se le asigna un alias anónimo (animal + adjetivo + emoji
de avatar) y un color; nadie en el chat sabe quién es quién: no hay registro, ni
nombres reales, ni emails, ni se muestran direcciones IP.

La identidad se ancla a una cookie de dispositivo persistente y se *deriva* de
ella por hash, no se guarda en memoria. Así el personaje aguanta el refresh,
cerrar el navegador, el reinicio del servidor y un redeploy.

Ejecutar:
    python chat_app.py
Luego abrir http://localhost:5000 en el navegador. Comparte esa URL (o el
código de sala) con la gente que quieras meter al chat.
"""

import hashlib
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from flask import Flask, g, jsonify, render_template, request

app = Flask(__name__)
# Ya no se usa para la identidad (va en nuestra propia cookie), pero se deja
# puesta para que cualquier uso futuro de session no quede sin firmar.
app.secret_key = os.environ.get("CHAT_SECRET_KEY", secrets.token_hex(32))

# --- Cookie de dispositivo ----------------------------------------------------
COOKIE = "chat_device"
COOKIE_DIAS = 400

# --- Vocabulario de alias -----------------------------------------------------
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
# La identidad NO depende de esto: se deriva de la cookie de dispositivo.
_lock = threading.Lock()
rooms = defaultdict(lambda: deque(maxlen=500))
_next_id = defaultdict(int)  # id incremental por sala, para el polling
_alias_en_uso = {}           # alias -> device_id, para no repetir entre presentes
_identidad = {}              # device_id -> (alias, emoji, color, pid), caché


def _device_id():
    """Id del dispositivo, tomado de nuestra cookie persistente."""
    if "device_id" not in g:
        actual = request.cookies.get(COOKIE, "")
        # Aceptamos solo el formato que emitimos nosotros; cualquier otra cosa
        # se descarta y se genera uno nuevo.
        if len(actual) == 32 and all(c in "0123456789abcdef" for c in actual):
            g.device_id = actual
        else:
            g.device_id = secrets.token_hex(16)
    return g.device_id


def _pid(device_id):
    """Id público del participante. Va en cada mensaje, así que es un hash:
    nunca se expone el valor de la cookie."""
    return hashlib.sha256(("pid:" + device_id).encode()).hexdigest()[:16]


def _derivar(device_id, offset=0):
    """Personaje que le toca a este dispositivo. Determinista: el mismo
    device_id da siempre el mismo alias, sin consultar nada guardado."""
    h = hashlib.sha256("{}:{}".format(offset, device_id).encode()).digest()
    animal, emoji, genero = ANIMALES[h[0] % len(ANIMALES)]
    adjetivo = concordar(ADJETIVOS[h[1] % len(ADJETIVOS)], genero)
    color = COLORES[h[2] % len(COLORES)]
    return "{} {}".format(animal, adjetivo), emoji, color


def identidad():
    """(alias, emoji, color, pid) del visitante actual."""
    did = _device_id()
    with _lock:
        cacheada = _identidad.get(did)
    if cacheada:
        return cacheada

    # Primera vez que vemos este dispositivo. Empezamos por el alias que le toca
    # y solo nos corremos si otro presente ya lo tiene, para que no haya dos
    # "Zorro Sereno" en la misma sala.
    for offset in range(40):
        nombre, emoji, color = _derivar(did, offset)
        with _lock:
            dueño = _alias_en_uso.get(nombre)
            if dueño is None or dueño == did:
                # No hay presencia, así que no sabemos cuándo alguien se va:
                # vaciamos el registro si crece demasiado en vez de agotarlo.
                if len(_alias_en_uso) > 600:
                    _alias_en_uso.clear()
                    _identidad.clear()
                _alias_en_uso[nombre] = did
                ident = (nombre, emoji, color, _pid(did))
                _identidad[did] = ident
                return ident

    # 40 combinaciones ocupadas: aceptamos repetir antes que dar error.
    nombre, emoji, color = _derivar(did, 40)
    ident = (nombre, emoji, color, _pid(did))
    with _lock:
        _identidad[did] = ident
    return ident


@app.after_request
def guardar_cookie(resp):
    """Graba (o renueva) la cookie de dispositivo en cada respuesta que haya
    resuelto una identidad. Al renovarla, la caducidad se corre hacia adelante
    mientras la persona siga usando el chat."""
    did = g.get("device_id")
    if did:
        # Render termina el TLS por delante, así que miramos X-Forwarded-Proto
        # además de request.is_secure.
        seguro = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
        resp.set_cookie(
            COOKIE,
            did,
            max_age=COOKIE_DIAS * 24 * 3600,
            httponly=True,      # el JS de la página no necesita leerla
            samesite="Lax",
            secure=seguro,
        )
    return resp


def sala_actual():
    """Código de sala pedido (?room=xxx) o 'general' por defecto."""
    code = (request.args.get("room") or request.form.get("room") or "general").strip()
    # Solo caracteres seguros; evitamos que el código rompa nada.
    code = "".join(c for c in code if c.isalnum() or c in "-_")[:32]
    return code or "general"


@app.route("/")
def index():
    alias, emoji, color, pid = identidad()
    return render_template(
        "chat.html",
        room=sala_actual(),
        alias=alias,
        emoji=emoji,
        color=color,
        pid=pid,
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
    alias, emoji, color, pid = identidad()
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
            "alias": alias,
            "emoji": emoji,
            "color": color,
            "pid": pid,  # solo para que el navegador marque "tú"
            "text": texto,
            # Epoch en ms: el navegador lo formatea en la zona horaria de cada
            # quien (el servidor de Render corre en UTC).
            "t": int(time.time() * 1000),
        }
        rooms[room].append(msg)
    return jsonify(ok=True, message=msg)


@app.route("/nueva-identidad", methods=["POST"])
def nueva_identidad():
    """Rota la cookie de dispositivo: se reaparece como otro anónimo."""
    viejo = _device_id()
    with _lock:
        anterior = _identidad.pop(viejo, None)
        if anterior:
            _alias_en_uso.pop(anterior[0], None)
    # after_request escribe este id nuevo en la cookie.
    g.device_id = secrets.token_hex(16)
    alias, emoji, color, pid = identidad()
    return jsonify(ok=True, alias=alias, emoji=emoji, color=color, pid=pid)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
