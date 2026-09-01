"""
Chat de grupo anónimo (con tarjetas, reactions y replies).

Cualquiera entra a una sala con solo abrir el enlace. NO hay ningún tipo de
identificación visible: nadie ve nombres, alias, emails ni IPs, y no se puede
saber quién mandó cada mensaje. El servidor solo necesita distinguir "tus
mensajes" de los demás y aplicar los castigos de las tarjetas, así que usa un
pid oculto: un hash de la cookie de dispositivo persistente (nunca se expone
el valor de la cookie en sí). Al no depender de Flask session, ese pid
sobrevive un reinicio del servidor o un redeploy.

Mecánicas:
- Reactions estilo WhatsApp (visibles como conteo, sin revelar quién reaccionó).
- 🟨 Tarjeta amarilla: cada persona tiene 3 cada 5 min (ventana fija). Si la
  mayoría de los online (>50%, sin contar al autor, con piso de 3 votos) le da
  tarjeta amarilla a un mensaje, este pasa a 🔴 roja y su autor no puede escribir
  por 1 minuto.
- 🥇 Tarjeta dorada: si la mayoría la da, el mensaje queda destacado.
- Replies (citas), presencia (online count), aviso de entrar/salir y typing.

El mute y el presupuesto de tarjetas son evadibles borrando la cookie del
dispositivo (o modo incógnito): sin login no hay forma real de impedirlo. Es
moderación entre pares / juego social, no un sistema de baneo real.

Ejecutar:
    python chat_app.py
Luego abrir http://localhost:5000 en el navegador.
"""

import hashlib
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from flask import Flask, g, jsonify, render_template, request

app = Flask(__name__)
# No se usa para identidad (va en la cookie de dispositivo propia), pero se
# deja puesta por si algo futuro necesita una session firmada.
app.secret_key = os.environ.get("CHAT_SECRET_KEY", secrets.token_hex(32))

# --- Cookie de dispositivo (identidad oculta y persistente) -------------------
COOKIE = "chat_device"
COOKIE_DIAS = 400

# --- Parámetros del juego -----------------------------------------------------
CARD_LIMIT = 3          # tarjetas amarillas por ventana
CARD_WINDOW = 300       # segundos (5 min) para recuperar las tarjetas
MUTE_SECONDS = 60       # castigo de la tarjeta roja
ONLINE_TTL = 15         # segundos sin sondear para seguir "en línea"
TYPING_TTL = 4          # segundos que dura el indicador de "escribiendo"
MIN_VOTES = 3           # piso de votos para roja/dorada
REACTIONS = {"❤️", "😂", "👍", "😮", "😢", "🔥", "🟨", "🥇"}

# --- Estado en memoria (requiere UN SOLO worker; ver Procfile) ----------------
_lock = threading.Lock()
rooms = defaultdict(lambda: deque(maxlen=500))   # room -> deque de mensajes
_next_id = defaultdict(int)                       # room -> id incremental
presence = defaultdict(dict)                      # room -> {pid: last_seen}
announced = defaultdict(set)                      # room -> pids ya anunciados online
typing = defaultdict(dict)                        # room -> {pid: last_typing}
budgets = {}                                       # pid -> {"start": t, "left": n}
mutes = {}                                         # pid -> epoch hasta cuando


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


def mi_pid():
    """Id oculto del participante actual: hash de su cookie de dispositivo,
    nunca el valor de la cookie en sí."""
    return hashlib.sha256(("pid:" + _device_id()).encode()).hexdigest()[:16]


@app.after_request
def guardar_cookie(resp):
    """Graba (o renueva) la cookie de dispositivo en cada respuesta que la
    haya resuelto. Al renovarla, la caducidad se corre hacia adelante mientras
    la persona siga usando el chat."""
    did = g.get("device_id")
    if did:
        # Render termina el TLS por delante, así que miramos X-Forwarded-Proto
        # además de request.is_secure.
        seguro = request.is_secure or request.headers.get("X-Forwarded-Proto") == "https"
        resp.set_cookie(
            COOKIE,
            did,
            max_age=COOKIE_DIAS * 24 * 3600,
            httponly=True,
            samesite="Lax",
            secure=seguro,
        )
    return resp


def sala_actual():
    """Código de sala pedido (?room=xxx) o 'general' por defecto."""
    code = (request.args.get("room") or request.form.get("room") or "general").strip()
    code = "".join(c for c in code if c.isalnum() or c in "-_")[:32]
    return code or "general"


# --- Helpers de estado --------------------------------------------------------
def online_pids(room, now):
    return {p for p, t in presence[room].items() if now - t <= ONLINE_TTL}


def _budget(pid, now):
    b = budgets.get(pid)
    if b is None or now - b["start"] >= CARD_WINDOW:
        b = {"start": now, "left": CARD_LIMIT}
        budgets[pid] = b
    return b


def cards_left(pid, now):
    return _budget(pid, now)["left"]


def append_system(room, texto):
    _next_id[room] += 1
    rooms[room].append({"id": _next_id[room], "kind": "system", "text": texto, "ts": ""})


def reaches_majority(room, msg, voters, now):
    online_excl = online_pids(room, now) - {msg["author_pid"]}
    v = len(voters)
    return v >= MIN_VOTES and v > 0.5 * len(online_excl)


def serialize(msg, pid):
    if msg["kind"] == "system":
        return {"id": msg["id"], "kind": "system", "text": msg["text"]}
    counts = {e: len(s) for e, s in msg["reactions"].items() if s}
    mine = [e for e, s in msg["reactions"].items() if pid in s]
    return {
        "id": msg["id"],
        "kind": "msg",
        "text": msg["text"],
        "ts": msg["ts"],
        "is_mine": msg["author_pid"] == pid,
        "reply": msg.get("reply"),
        "reactions": counts,
        "mine": mine,
        "status": msg["status"],
    }


# --- Rutas --------------------------------------------------------------------
@app.route("/")
def index():
    mi_pid()  # resuelve/crea la cookie de dispositivo para esta respuesta
    return render_template("chat.html", room=sala_actual())


@app.route("/state")
def state():
    """Sondeo principal: mensajes nuevos desde ?since=<id> + estado vivo
    (reactions/estatus de mensajes recientes, en línea, typing, mute, tarjetas
    y el destacado dorado)."""
    pid = mi_pid()
    room = sala_actual()
    now = time.time()
    try:
        since = int(request.args.get("since", 0))
    except ValueError:
        since = 0

    with _lock:
        presence[room][pid] = now

        # Anuncios de entrar / salir (heartbeat en el sondeo principal).
        online = online_pids(room, now)
        for _ in (online - announced[room]):
            append_system(room, "🟢 Alguien entró a la sala")
        for _ in (announced[room] - online):
            append_system(room, "⚪ Alguien salió de la sala")
        announced[room] = set(online)

        dq = rooms[room]
        nuevos = [serialize(m, pid) for m in dq if m["id"] > since]
        # Estado mutable (reactions/estatus) de los últimos mensajes, para que
        # el cliente actualice mensajes viejos cuando reciben reacciones.
        upd = []
        for m in list(dq)[-80:]:
            if m["kind"] != "msg":
                continue
            upd.append({
                "id": m["id"],
                "reactions": {e: len(s) for e, s in m["reactions"].items() if s},
                "mine": [e for e, s in m["reactions"].items() if pid in s],
                "status": m["status"],
            })

        # Destacado (última dorada) para el banner superior.
        gold = None
        for m in reversed(dq):
            if m["kind"] == "msg" and m["status"] == "gold":
                gold = {"id": m["id"], "text": m["text"]}
                break

        typing_others = sum(
            1 for p, t in typing[room].items() if p != pid and now - t <= TYPING_TTL
        )
        muted = max(0, int(mutes.get(pid, 0) - now))
        cards = cards_left(pid, now)

    return jsonify(
        new=nuevos, upd=upd, online=len(online), typing=typing_others,
        muted=muted, cards=cards, gold=gold,
    )


@app.route("/send", methods=["POST"])
def send():
    pid = mi_pid()
    room = sala_actual()
    now = time.time()
    texto = (request.form.get("text") or "").strip()
    if not texto:
        return jsonify(ok=False, error="mensaje vacío"), 400
    texto = texto[:2000]

    with _lock:
        presence[room][pid] = now
        restante = int(mutes.get(pid, 0) - now)
        if restante > 0:
            return jsonify(ok=False, muted=restante), 403

        reply = None
        try:
            rid = int(request.form.get("reply_to", ""))
            for m in rooms[room]:
                if m["id"] == rid and m["kind"] == "msg":
                    reply = {"id": rid, "text": m["text"][:120]}
                    break
        except (ValueError, TypeError):
            pass

        _next_id[room] += 1
        msg = {
            "id": _next_id[room],
            "kind": "msg",
            "author_pid": pid,
            "text": texto,
            "ts": time.strftime("%H:%M"),
            "reply": reply,
            "reactions": defaultdict(set),
            "status": None,
        }
        rooms[room].append(msg)
        typing[room].pop(pid, None)
    return jsonify(ok=True, id=msg["id"])


@app.route("/react", methods=["POST"])
def react():
    pid = mi_pid()
    room = sala_actual()
    now = time.time()
    emoji = request.form.get("emoji", "")
    if emoji not in REACTIONS:
        return jsonify(ok=False, error="reacción inválida"), 400
    try:
        mid = int(request.form.get("id", ""))
    except (ValueError, TypeError):
        return jsonify(ok=False, error="id inválido"), 400

    with _lock:
        presence[room][pid] = now
        msg = next((m for m in rooms[room] if m["id"] == mid and m["kind"] == "msg"), None)
        if msg is None:
            return jsonify(ok=False, error="mensaje no encontrado"), 404

        es_tarjeta = emoji in ("🟨", "🥇")
        if es_tarjeta and msg["author_pid"] == pid:
            return jsonify(ok=False, error="no puedes votar tu propio mensaje"), 400

        votos = msg["reactions"][emoji]
        if pid in votos:
            # Quitar reacción. Si era tarjeta amarilla y aún no era roja, devolvemos la tarjeta.
            votos.discard(pid)
            if emoji == "🟨" and msg["status"] != "red":
                b = _budget(pid, now)
                b["left"] = min(CARD_LIMIT, b["left"] + 1)
        else:
            if emoji == "🟨":
                b = _budget(pid, now)
                if b["left"] <= 0:
                    return jsonify(ok=False, error="sin tarjetas", cards=0), 429
                b["left"] -= 1
            votos.add(pid)
            # ¿Alcanza mayoría para roja / dorada?
            if msg["status"] is None and es_tarjeta and reaches_majority(room, msg, votos, now):
                if emoji == "🟨":
                    msg["status"] = "red"
                    mutes[msg["author_pid"]] = now + MUTE_SECONDS
                    append_system(room, "🔴 Un mensaje recibió tarjeta roja")
                else:
                    msg["status"] = "gold"
                    append_system(room, "🥇 Un mensaje fue destacado por el grupo")

        cards = cards_left(pid, now)
    return jsonify(ok=True, cards=cards)


@app.route("/typing", methods=["POST"])
def typing_ping():
    pid = mi_pid()
    room = sala_actual()
    now = time.time()
    with _lock:
        presence[room][pid] = now
        typing[room][pid] = now
    return jsonify(ok=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
