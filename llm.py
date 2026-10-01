#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""llm.py — логика servstart без GTK (тестируемая часть).

Слои:
  - каталог бэкендов (backends.json — справочник из сети для сопоставления);
  - discovery: поиск бинарников в PATH/известных путях и моделей *.gguf по корням;
  - чтение GGUF-заголовка (general.architecture / general.name) для сопоставления;
  - конфиг пользователя (~/.config/servstart/config.json), атомарная запись;
  - сборка argv из шаблона бэкенда + настроек;
  - запуск/остановка/статус/метрики (VRAM/RAM).

Никакого GTK — этот модуль импортируется и из трея, и из юнит-тестов.
"""
import glob
import json
import os
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ----------------------------------------------------------------------------
# Публичные структуры
# ----------------------------------------------------------------------------


class Setting:
    """Схема одной настройки запуска (из каталога)."""

    def __init__(self, d):
        self.key = d["key"]
        self.type = d.get("type", "str")
        self.label = d.get("label", self.key)
        self.what = d.get("what", "")
        self.why = d.get("why", "")
        self.default = d.get("default")
        self.min = d.get("min")
        self.max = d.get("max")
        self.step = d.get("step", 1)
        self.choices = d.get("choices")
        self.on_value = d.get("on_value", "on")
        self.off_value = d.get("off_value", "off")
        self.split = bool(d.get("split", False))


class Backend:
    """Один бэкенд из каталога + найденный бинарник."""

    def __init__(self, d):
        self.id = d["id"]
        self.name = d.get("name", self.id)
        self.description = d.get("description", "")
        self.provider = d.get("provider", self.id)
        self.binaries = d.get("binaries", [])
        self.search_paths = [os.path.expanduser(p) for p in d.get("search_paths", [])]
        self.formats = d.get("formats", [])
        self.arch = d.get("arch")                 # None = любой (кроме exclude_arch)
        self.exclude_arch = set(d.get("exclude_arch", []))
        self.health = d.get("health", "/health")
        self.models_route = d.get("models_route", "/v1/models")
        self.port_default = d.get("port_default", 8080)
        self.env = d.get("env", {})
        self.argv_template = d.get("argv_template", [])
        self.wired = d.get("wired", True)         # False = справочная запись, запуск не готов
        self.settings = [Setting(s) for s in d.get("settings", [])]
        self.setting_map = {s.key: s for s in self.settings}
        self.binary_path = None                   # заполняется discovery
        self.binary_name = None


class Model:
    """Найденная модель (основной файл GGUF) + вспомогательные файлы."""

    def __init__(self, path, name, arch, size_bytes, backend_id, aux=None):
        self.path = path
        self.name = name or os.path.basename(path)
        self.arch = arch
        self.size_bytes = size_bytes
        self.backend_id = backend_id
        self.aux = aux or {}     # mmproj / mtp / draft — пути к вспомогательным GGUF
        self.alias = name or os.path.basename(path)


# ----------------------------------------------------------------------------
# GGUF-заголовок
# ----------------------------------------------------------------------------

_SCALAR_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _read_gguf_value(f, t):
    if t == 8:  # string
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")
    if t == 7:  # bool
        return bool(f.read(1)[0])
    if t == 9:  # array
        (atype,) = struct.unpack("<I", f.read(4))
        (alen,) = struct.unpack("<Q", f.read(8))
        out = []
        if atype == 8:
            for _ in range(alen):
                (n,) = struct.unpack("<Q", f.read(8))
                out.append(f.read(n).decode("utf-8", "replace"))
        else:
            for _ in range(alen):
                out.append(_read_gguf_value(f, atype))
        return out
    if t in (6, 12):  # f32 / f64
        sz = 4 if t == 6 else 8
        return struct.unpack("<" + ("f" if sz == 4 else "d"), f.read(sz))[0]
    if t == 1:
        return struct.unpack("<b", f.read(1))[0]
    if t == 3:
        return struct.unpack("<h", f.read(2))[0]
    if t == 5:
        return struct.unpack("<i", f.read(4))[0]
    if t == 11:
        return struct.unpack("<q", f.read(8))[0]
    sz = _SCALAR_SZ[t]
    return struct.unpack("<" + {1: "B", 2: "H", 4: "I", 8: "Q"}[sz], f.read(sz))[0]


def read_gguf_meta(path, keys=("general.architecture", "general.name",
                                "general.type", "general.quantization_version")):
    """Читает только заголовок GGUF (metadata), не трогая тензоры."""
    meta = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                meta["_error"] = "not GGUF"
                return meta
            meta["gguf_version"], = struct.unpack("<I", f.read(4))
            meta["tensor_count"], = struct.unpack("<Q", f.read(8))
            (nk,) = struct.unpack("<Q", f.read(8))
            for _ in range(nk):
                (klen,) = struct.unpack("<Q", f.read(8))
                key = f.read(klen).decode("utf-8", "replace")
                (t,) = struct.unpack("<I", f.read(4))
                v = _read_gguf_value(f, t)
                if key in keys:
                    meta[key] = v
    except Exception as e:
        meta["_error"] = "%s: %s" % (type(e).__name__, e)
    return meta


# ----------------------------------------------------------------------------
# Каталог
# ----------------------------------------------------------------------------


def load_catalog(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["id"]: Backend(b) for b in data.get("backends", [])}


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------


def _which_all(name):
    """Все вхождения бинарника в PATH."""
    found = []
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            found.append(p)
    return found


def discover_backends(catalog, extra_paths=None):
    """Возвращает список найденных бэкендов (с заполненным binary_path)."""
    found = []
    for b in catalog.values():
        candidates = []
        for name in b.binaries:
            candidates += _which_all(name)
        for d in b.search_paths:
            for name in b.binaries:
                p = os.path.join(d, name)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    candidates.append(p)
        for p in (extra_paths or []):
            if os.path.isfile(p) and os.access(p, os.X_OK):
                candidates.append(p)
        if candidates:
            b.binary_path = candidates[0]
            b.binary_name = os.path.basename(b.binary_path)
            found.append(b)
    return found


def _is_draft_dir(dirname):
    return dirname.endswith("-draft")


def find_models(roots):
    """Ищет *.gguf по корням, группирует по каталогу, выделяет main + aux.

    Возвращает список Model (backend_id пока None — сопоставление отдельным шагом).
    """
    files = []
    for root in roots:
        root = os.path.expanduser(root)
        for pat in (os.path.join(root, "**", "*.gguf"),
                    os.path.join(root, "**", "*.GGUF")):
            files += glob.glob(pat, recursive=True)
    files = sorted(set(files))

    by_dir = {}
    for f in files:
        d = os.path.dirname(f)
        if _is_draft_dir(d):
            by_dir.setdefault(d, []).append(f)
            continue
        # MTP/ — подкаталог, относим к родителю модели
        if os.path.basename(d) == "MTP":
            by_dir.setdefault(os.path.dirname(d), []).append(f)
            continue
        by_dir.setdefault(d, []).append(f)

    models = []
    draft_by_base = {}
    for d, fs in by_dir.items():
        if _is_draft_dir(d):
            biggest = max(fs, key=lambda p: os.path.getsize(p))
            draft_by_base[d[:-len("-draft")]] = biggest
            continue
        # main = *-merged.gguf, иначе самый большой файл
        merged = [p for p in fs if p.endswith("-merged.gguf")]
        if merged:
            main = max(merged, key=lambda p: os.path.getsize(p))
        else:
            main = max(fs, key=lambda p: os.path.getsize(p))
        rest = [p for p in fs if p != main]

        meta = read_gguf_meta(main)
        if meta.get("_error") or not meta.get("general.architecture"):
            continue
        # mmproj (clip) — не модель
        if meta.get("general.architecture") == "clip" or meta.get("general.type") == "mmproj":
            continue

        aux = {}
        for p in rest:
            base = os.path.basename(p)
            if base.lower().startswith("mmproj"):
                aux["mmproj"] = p
            elif "mtp" in base.lower() or os.path.basename(os.path.dirname(p)) == "MTP":
                aux.setdefault("mtp", p)
        models.append(Model(
            path=main,
            name=meta.get("general.name") or os.path.basename(main),
            arch=meta.get("general.architecture"),
            size_bytes=os.path.getsize(main),
            backend_id=None,
            aux=aux,
        ))

    for m in models:
        base = os.path.dirname(m.path)
        if base in draft_by_base:
            m.aux["draft"] = draft_by_base[base]
    return models


def match_model_to_backend(model, catalog):
    """Какой бэкенд в состоянии запустить модель (по architecture/формату)."""
    arch = model.arch
    for b in catalog.values():
        if b.arch is not None:
            if arch in b.arch:
                return b.id
            continue
        if arch in b.exclude_arch:
            continue
        if "gguf" in b.formats:
            return b.id
    return None


# ----------------------------------------------------------------------------
# Конфиг пользователя
# ----------------------------------------------------------------------------


def config_path():
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "servstart", "config.json")


def config_defaults(catalog):
    """Значения по умолчанию из каталога: {backend_id: {setting: value}}."""
    out = {}
    for bid, b in catalog.items():
        out[bid] = {s.key: s.default for s in b.settings}
        if b.port_default:
            # порт по умолчанию из каталога, если схема задаёт порт
            for s in b.settings:
                if s.key == "port":
                    out[bid]["port"] = b.port_default
    return out


def load_config(path=None):
    path = path or config_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(data, path=None):
    path = path or config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)  # атомарно


def effective_settings(backend, model, cfg):
    """default <- backend override <- model override."""
    s = dict(config_defaults({backend.id: backend})[backend.id])
    s.update((cfg.get("backends") or {}).get(backend.id, {}))
    s.update((cfg.get("models") or {}).get(model.path, {}))
    return s


# ----------------------------------------------------------------------------
# Сборка argv
# ----------------------------------------------------------------------------


def _derived(backend, model, s):
    out = {}
    gpu = str(s.get("gpu_layers", "999"))
    out["mtp_args"] = []
    if s.get("mtp") and model.aux.get("mtp"):
        out["mtp_args"] = ["-md", model.aux["mtp"], "--spec-type", "draft-mtp",
                           "--spec-draft-n-min", "0", "--spec-draft-n-max", "1",
                           "-ngld", gpu]
    out["mmproj_args"] = []
    if s.get("mmproj") and model.aux.get("mmproj"):
        out["mmproj_args"] = ["--mmproj", model.aux["mmproj"]]
    out["draft_args"] = []
    if s.get("draft") and model.aux.get("draft"):
        out["draft_args"] = ["--draft", model.aux["draft"]]
    return out


def build_argv(backend, model, cfg):
    """Собирает полную командную строку запуска из шаблона + настроек.

    Настройки передаются бэкенду как аргументы (требование ТЗ).
    """
    s = effective_settings(backend, model, cfg)
    derived = _derived(backend, model, s)
    out = []
    for tok in backend.argv_template:
        if tok.startswith("{") and tok.endswith("}"):
            key = tok[1:-1]
            if key == "binary":
                out.append(backend.binary_path)
            elif key == "model":
                out.append(model.path)
            elif key == "alias":
                out.append(model.alias)
            elif key in ("mtp_args", "mmproj_args", "draft_args"):
                out += derived[key]
            else:
                st = backend.setting_map.get(key)
                val = s.get(key, st.default if st else None)
                if st is not None and st.type == "bool":
                    out.append(st.on_value if val else st.off_value)
                elif st is not None and st.split and val:
                    out += shlex.split(str(val))
                else:
                    out.append("" if val is None else str(val))
        else:
            out.append(tok)
    return [t for t in out if t != ""]


# ----------------------------------------------------------------------------
# Память: VRAM/RAM (sysfs + /proc)
# ----------------------------------------------------------------------------


def _read_int(p):
    try:
        with open(p) as f:
            return int(f.read().strip())
    except Exception:
        return 0


def gpu_pool_usage():
    """{(vram_used,vram_total),(gtt_used,gtt_total)} в байтах, суммарно по картам."""
    vram_u = vram_t = gtt_u = gtt_t = 0
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        v = _read_int(os.path.join(card, "mem_info_vram_total"))
        g = _read_int(os.path.join(card, "mem_info_gtt_total"))
        if not (v or g):
            continue
        vram_t += v
        gtt_t += g
        vram_u += _read_int(os.path.join(card, "mem_info_vram_used"))
        gtt_u += _read_int(os.path.join(card, "mem_info_gtt_used"))
    return (vram_u, vram_t), (gtt_u, gtt_t)


def ram_usage():
    """(used, total) системной ОЗУ из /proc/meminfo, байты."""
    total = used = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
        used = total - avail
    except Exception:
        pass
    return used, total


def proc_rss(pid):
    """RSS процесса в байтах (VmRSS из /proc/<pid>/status)."""
    if not pid:
        return 0
    try:
        with open("/proc/%s/status" % pid) as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def fmt_gib(b):
    return "%.1f" % (b / 2 ** 30)


def memory_summary(pid=None):
    """Единая память (UMA) на Strix Halo: «RAM (UMA) total/used GiB». Предельно просто."""
    ru, rt = ram_usage()
    return "RAM (UMA) %.1f/%.1f GiB" % (rt / 2 ** 30, ru / 2 ** 30)


# ----------------------------------------------------------------------------
# HTTP-опрос сервера
# ----------------------------------------------------------------------------


def _get(url, timeout=3):
    req = urllib.request.Request(url, headers={"User-Agent": "servstart/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def health_ok(host, port):
    try:
        _get("http://%s:%s/health" % (host, port), timeout=3)
        return True
    except Exception:
        return False


def served_model_name(host, port):
    try:
        d = json.loads(_get("http://%s:%s/v1/models" % (host, port), timeout=5))
        return (d.get("data") or [{}])[0].get("id")
    except Exception:
        return None


def port_in_use(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def find_pid_by_port(port):
    """pid процесса, слушающего TCP-порт (через ss), либо None.

    Надёжнее pid-файла: работает, даже если pid-файл утерян/устарел.
    """
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp"], stderr=subprocess.DEVNULL, timeout=5).decode()
    except Exception:
        return None
    for line in out.splitlines():
        if ":%d " % port not in line:
            continue
        # строка вида: ... users:(("luce_server",pid=400520,fd=6))
        for tok in line.replace("(", " ").replace(")", " ").replace(",", " ").split():
            if tok.startswith("pid="):
                try:
                    return int(tok[4:])
                except ValueError:
                    pass
    return None


# ----------------------------------------------------------------------------
# Запуск / остановка
# ----------------------------------------------------------------------------

CHOOM = None


def _choom():
    global CHOOM
    if CHOOM is None:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, "choom")
            if os.path.isfile(p) and os.access(p, os.X_OK):
                CHOOM = p
                break
        else:
            CHOOM = ""
    return CHOOM


def start_backend(backend, model, cfg, logfile, pidfile):
    """Запускает модель выбранным бэкендом. Возвращает (pid, error).

    Запуск — setsid + nohup, stdout/stderr в logfile, stdin в /dev/null.
    При нехватке памяти ядро убьёт сервер первым (choom -n 1000), а не рабочий стол.
    """
    argv = build_argv(backend, model, cfg)
    env = dict(os.environ)
    env.update(backend.env)
    cmd = []
    c = _choom()
    if c:
        cmd += [c, "-n", "1000", "--"]
    cmd += argv

    try:
        log = open(logfile, "ab")
    except OSError as e:
        return None, "не могу открыть лог %s: %s" % (logfile, e)
    try:
        with open(os.devnull, "rb") as devnull:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                stdin=devnull, start_new_session=True, close_fds=True)
    except OSError as e:
        log.close()
        return None, "не удалось запустить %s: %s" % (backend.binary_path, e)
    log.close()

    pid = proc.pid
    with open(pidfile, "w") as f:
        f.write(str(pid))
    return pid, None


def wait_ready(host, port, pid, timeout_s, poll_s=3):
    """Ждёт /health; возвращает (ok, err). Если процесс умер — err с хвостом лога."""
    waited = 0
    while waited < timeout_s:
        if pid and not _pid_alive(pid):
            return False, "процесс завершился при загрузке (pid %s)" % pid
        if health_ok(host, port):
            return True, None
        time.sleep(poll_s)
        waited += poll_s
    return False, "не ответил на /health за %ds" % timeout_s


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def read_pid(pidfile):
    try:
        with open(pidfile) as f:
            return int(f.read().strip())
    except Exception:
        return None


def pid_alive(pid):
    """Публичная проверка живости процесса."""
    return _pid_alive(pid)


def ensure_stopped(pid, timeout_s=300):
    """Добить процесс SIGKILL, пока не умрёт.

    Нужно для D-состояния (uninterruptible IO): пока модель грузит 90+ GiB,
    процесс не отвечает даже на SIGKILL — сигнал ядро применяет после выхода из IO.
    """
    deadline = time.monotonic() + timeout_s
    while pid and _pid_alive(pid) and time.monotonic() < deadline:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(3)
    return (not _pid_alive(pid)) if pid else True


def stop_backend(pidfile, pid=None):
    """Мягкая остановка (SIGTERM -> SIGKILL). Возвращает True, если процесс остановлен.

    pid-файл удаляется только когда процесса больше нет — иначе при неудачной
    остановке (процесс в uninterruptible IO) теряли pid и не могли убить его позже.
    """
    pid = pid or read_pid(pidfile)
    if pid and _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(20):
            if not _pid_alive(pid):
                break
            time.sleep(0.5)
        if _pid_alive(pid):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            time.sleep(2)
    stopped = bool(pid) and not _pid_alive(pid)
    if not pid or not _pid_alive(pid):
        if os.path.exists(pidfile):
            try:
                os.remove(pidfile)
            except OSError:
                pass
    return stopped


# ----------------------------------------------------------------------------
# Preflight
# ----------------------------------------------------------------------------


def preflight(backend, model, cfg):
    """Список сообщений (level, text): ok/warn/error. Пусто по errors => можно запускать."""
    msgs = []
    if not backend:
        msgs.append(("error", "бэкенд не найден"))
        return msgs
    if not backend.binary_path:
        msgs.append(("error", "нет бинарника %s (искал в %s)"
                     % ("/".join(backend.binaries), ", ".join(backend.search_paths) or "PATH")))
    else:
        msgs.append(("ok", "бинарник: %s" % backend.binary_path))
    if not os.path.isfile(model.path):
        msgs.append(("error", "нет файла модели: %s" % model.path))
    else:
        msgs.append(("ok", "модель: %s (%.2f GiB)" % (os.path.basename(model.path),
                                                      model.size_bytes / 2 ** 30)))
    s = effective_settings(backend, model, cfg)
    port = int(s.get("port", backend.port_default))
    if port_in_use(port) and not health_ok("127.0.0.1", port):
        msgs.append(("error", "порт %d занят другим процессом" % port))
    # память: оценка по размеру модели
    (vu, vt), (gu, gt) = gpu_pool_usage()
    avail_gpu = (vt + gt - vu - gu) / 2 ** 30
    need = model.size_bytes / 2 ** 30 + 2.0  # +headroom
    if avail_gpu < need:
        msgs.append(("warn", "GPU-пул %.1f GiB, модели нужно ~%.1f GiB — может не влезть"
                     % (avail_gpu, need)))
    return msgs
