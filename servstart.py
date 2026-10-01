#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""servstart — трей-сервис управления локальными LLM-бэкендами и моделями.

После запуска автоматически ищет в системе доступные бэкенды (llama.cpp,
Lucebox и др. — по справочнику backends.json) и модели (*.gguf), сопоставляет
их по architecture из GGUF-заголовка и строит каскадное меню в системном трее:
бэкенд -> модели (только те, которые бэкенд в состоянии запустить).

Иконка-«светофор»: серый — ничего не запущено, зелёный — модель работает,
красный — сбой/ошибка. В статусе — имя модели, адрес эндпоинта (OpenAI) и
прогресс загрузки/останова (секундомер).

Окна (настройки / логи / готовность) ведут себя как всплывающие: при выборе
пункта все прочие окна закрываются, выбранное поднимается наверх; клик вне
окна (или по иконке трея) его убирает; изменения применяются только по «ОК».

Позиционирование: XWayland (GDK_BACKEND=x11). Явный GDK_SCALE не задаём.
Вся логика (без GTK) — в llm.py, покрыта юнит-тестами.
"""
import os

# До импорта GTK: XWayland, иначе move() игнорируется на нативном Wayland.
os.environ.setdefault("GDK_BACKEND", "x11")

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, Gdk, GLib

try:
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3
    HAS_INDICATOR = True
except Exception:
    AyatanaAppIndicator3 = None
    HAS_INDICATOR = False

import os
import subprocess
import threading
import time

import llm

# ----------------------------------------------------------------------------
# Палитра (тёмная)
# ----------------------------------------------------------------------------
BG = "#1a1a1a"
SURFACE = "#2a2a2a"
SURFACE_ALT = "#333333"
TEXT = "#e0e0e0"
TEXT_DIM = "#8a8a8a"
BORDER = "#444444"
ACCENT = "#d0702f"
GREEN = "#30c53a"
RED = "#e03a3a"

CSS = """
window { background-color: %(bg)s; color: %(text)s; }
frame > border { border-color: %(border)s; }
frame > label { color: %(text_dim)s; font-weight: bold; }
button {
    background-image: none; background-color: %(surface)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: 4px; padding: 4px 10px;
}
button:hover { background-color: %(surface_alt)s; }
button:active { background-color: %(accent)s; color: white; }
button.suggested-action { background-color: %(accent)s; color: white; }
entry, spinbutton, combobox window {
    background-color: %(surface)s; color: %(text)s; border: 1px solid %(border)s;
}
combobox window { background-color: %(surface)s; }
scale trough { background-color: %(surface)s; }
scale highlight { background-color: %(accent)s; }
.hint { color: %(text_dim)s; font-size: 0.85em; }
.title { color: %(text)s; font-weight: bold; }
""" % {"bg": BG, "text": TEXT, "surface": SURFACE, "surface_alt": SURFACE_ALT,
       "border": BORDER, "accent": ACCENT, "text_dim": TEXT_DIM}


def _icon_file(color):
    cand = [
        os.path.expanduser("~/.local/share/icons/servstart-%s.png" % color),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "icons", "servstart-%s.png" % color),
    ]
    for c in cand:
        if os.path.isfile(c):
            return c
    return None


def _tail(path, n=25):
    try:
        with open(path, "r", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------
# Приложение
# ----------------------------------------------------------------------------


class ServStart:
    def __init__(self, catalog_path=None):
        catalog_path = catalog_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "backends.json")
        self.catalog = llm.load_catalog(catalog_path)
        self.cfg = llm.load_config()

        # discovery
        self.backends = llm.discover_backends(self.catalog)
        self.models = llm.find_models(["/mnt/models", "~/models",
                                       "~/.cache/ollama/models",
                                       "~/.local/share/ollama"])
        for m in self.models:
            m.backend_id = llm.match_model_to_backend(m, self.catalog)
        self.by_backend = {}
        for m in self.models:
            self.by_backend.setdefault(m.backend_id, []).append(m)

        # состояние
        self.running = set()      # paths запущенных моделей
        self.error_state = None   # сообщение последнего сбоя
        self._starting = {}       # path -> time.monotonic() (идёт загрузка)
        self._stopping = {}       # path -> time.monotonic() (идёт остановка)
        self._cancelled = set()   # модели, остановленные во время загрузки

        # UI
        self._apply_css()
        self.win_settings = None
        self.win_logs = None
        self.win_preflight = None
        self.indicator = None
        self.status_item = None
        self.mem_item = None
        self.settings_item = None   # пункт «Стартовая настройка…»
        self.preflight_item = None  # пункт «Проверить готовность»
        self.model_items = {}     # path -> Gtk.MenuItem (пункты моделей)
        self.backend_items = {}   # id -> Gtk.MenuItem (пункты бэкендов)
        self._widgets = {}
        self._log_state = {}      # logfile -> {pos, inode, partial} (инкрементальное чтение)

        if HAS_INDICATOR:
            self.build_indicator()
        self.build_settings_window()
        self.build_logs_window()
        self.build_preflight_window()

        GLib.timeout_add_seconds(3, self.tick)
        GLib.timeout_add_seconds(1, self._status_tick)
        self.tick()

    # ------------------------------------------------------------------ css
    def _apply_css(self):
        settings = Gtk.Settings.get_default()
        settings.set_property("gtk-application-prefer-dark-theme", True)
        prov = Gtk.CssProvider()
        prov.load_from_data(CSS.encode())
        Gtk.StyleContext.add_provider_for_screen(
            Gdk.Screen.get_default(), prov, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

    # ------------------------------------------------------------- indicator
    def build_indicator(self):
        gray = _icon_file("gray") or "servstart-gray"
        self.indicator = AyatanaAppIndicator3.Indicator.new(
            "servstart", gray, AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS)
        self.indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)

        menu = Gtk.Menu()
        self.status_item = Gtk.MenuItem(label="…")
        self.status_item.set_sensitive(False)
        menu.append(self.status_item)
        self.mem_item = Gtk.MenuItem(label="")
        self.mem_item.set_sensitive(False)
        menu.append(self.mem_item)
        menu.append(Gtk.SeparatorMenuItem())

        # каскад: бэкенд -> модели
        for b in self.backends:
            bi = Gtk.MenuItem(label=b.name)
            self.backend_items[b.id] = bi
            sub = Gtk.Menu()
            ms = self.by_backend.get(b.id, [])
            if not ms:
                di = Gtk.MenuItem(label="нет совместимых моделей")
                di.set_sensitive(False)
                sub.append(di)
            for m in ms:
                mi = Gtk.MenuItem(label=m.name)
                mi.connect("activate", self._on_model_toggle, b, m)
                self.model_items[m.path] = mi
                sub.append(mi)
            bi.set_submenu(sub)
            menu.append(bi)

        if not self.backends:
            nb = Gtk.MenuItem(label="бэкенды не найдены")
            nb.set_sensitive(False)
            menu.append(nb)

        menu.append(Gtk.SeparatorMenuItem())

        s_item = Gtk.MenuItem(label="Стартовая настройка…")
        s_item.connect("activate", lambda *a: self.show_settings())
        self.settings_item = s_item
        menu.append(s_item)
        l_item = Gtk.MenuItem(label="Логи…")
        l_item.connect("activate", lambda *a: self.show_logs())
        menu.append(l_item)
        c_item = Gtk.MenuItem(label="Проверить готовность")
        c_item.connect("activate", lambda *a: self.show_preflight())
        self.preflight_item = c_item
        menu.append(c_item)
        menu.append(Gtk.SeparatorMenuItem())
        stop_item = Gtk.MenuItem(label="Остановка сервиса")
        stop_item.connect("activate", lambda *a: self._on_stop_service())
        menu.append(stop_item)
        q_item = Gtk.MenuItem(label="Выход")
        q_item.connect("activate", lambda *a: self._on_quit())
        menu.append(q_item)
        menu.show_all()

        self.indicator.set_menu(menu)

    # ------------------------------------------------------------- состояние
    def _model_host_port(self, b, m):
        s = llm.effective_settings(b, m, self.cfg)
        return s.get("host", "127.0.0.1"), int(s.get("port", b.port_default))

    def _log_pid_files(self, m):
        d = os.path.dirname(m.path)
        return os.path.join(d, "server.log"), os.path.join(d, "server.pid")

    def _log_event(self, m, msg):
        """Дописать [timestamp] в лог модели (события servstart)."""
        logfile, _ = self._log_pid_files(m)
        try:
            with open(logfile, "a", encoding="utf-8") as f:
                f.write("[%s] servstart: %s\n"
                        % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))
        except OSError:
            pass

    def _backend_of(self, m):
        return next((b for b in self.backends if b.id == m.backend_id), None)

    def _name_of(self, path):
        m = next((x for x in self.models if x.path == path), None)
        return m.name if m else os.path.basename(path)

    def tick(self):
        new_running = set()
        for b in self.backends:
            for m in self.by_backend.get(b.id, []):
                host, port = self._model_host_port(b, m)
                if llm.health_ok(host, port):
                    new_running.add(m.path)
        self.running = new_running
        self._refresh_indicator()
        return True

    def _running_status_lines(self):
        lines = []
        for path in sorted(self.running):
            m = next((x for x in self.models if x.path == path), None)
            if not m:
                continue
            b = self._backend_of(m)
            host, port = self._model_host_port(b, m) if b else ("127.0.0.1", "?")
            lines.append("%s — http://%s:%s/v1" % (m.name, host, port))
        return lines

    def _status_text(self):
        now = time.monotonic()
        # остановка приоритетнее: пользователь явно попросил остановить
        if self._stopping:
            parts = ["останавливаю %s… %ds" % (self._name_of(p), int(now - t))
                     for p, t in self._stopping.items()]
            return " · ".join(parts)
        if self._starting:
            parts = ["запускаю %s… %ds" % (self._name_of(p), int(now - t))
                     for p, t in self._starting.items()]
            return " · ".join(parts)
        if self.running:
            return "работает: " + " | ".join(self._running_status_lines())
        if self.error_state:
            return "СБОЙ: " + self.error_state
        return "ничего не запущено"

    def _status_tick(self):
        # секундомер загрузки/останова обновляется чаще, чем полный опрос
        if self._starting or self._stopping:
            if self.status_item:
                self.status_item.set_label(self._status_text())
        return True

    def _refresh_indicator(self):
        if not self.indicator:
            return
        if self.running:
            color, desc = "green", "запущено"
        elif self.error_state:
            color, desc = "red", "сбой"
        else:
            color, desc = "gray", "не запущено"
        p = _icon_file(color)
        if p:
            self.indicator.set_icon_full(p, desc)

        mem = llm.memory_summary()
        status = self._status_text()

        if self.status_item:
            self.status_item.set_label(status)
        if self.mem_item:
            self.mem_item.set_label(mem)
        for path, mi in self.model_items.items():
            name = self._name_of(path)
            mi.set_label(("✓ " if path in self.running else "") + name)
        # пока модель работает/грузится/останавливается — выбор и настройки недоступны
        busy = bool(self.running) or bool(self._starting) or bool(self._stopping)
        for mi in self.model_items.values():
            mi.set_sensitive(not busy)
        for bi in self.backend_items.values():
            bi.set_sensitive(not busy)
        if self.settings_item:
            self.settings_item.set_sensitive(not busy)
        if self.preflight_item:
            self.preflight_item.set_sensitive(not busy)
        self.indicator.set_title("servstart — " + status)

    # ------------------------------------------------------------- запуск/стоп
    def _on_model_toggle(self, item, b, m):
        print("[%s] toggle %s (running=%s)"
              % (time.strftime("%H:%M:%S"), m.name, m.path in self.running),
              flush=True)
        if m.path in self._starting:
            return
        if m.path in self.running:
            self._stop_model(b, m)
        else:
            self._start_model(b, m)

    def _stop_model(self, b, m, silent=False):
        _, pidfile = self._log_pid_files(m)
        pid = llm.read_pid(pidfile)
        if not pid:
            # pid-файл утерян — ищем процесс по порту
            host, port = self._model_host_port(b, m)
            pid = llm.find_pid_by_port(port)
        # остановка во время загрузки: снимаем «запускаю», не показываем ошибку потом
        if m.path in self._starting:
            self._cancelled.add(m.path)
            self._starting.pop(m.path, None)
        if not pid:
            self._stopping.pop(m.path, None)
            self.error_state = None
            self.tick()
            if m.path in self.running and not silent:
                self._dialog("Остановка",
                             "%s запущена, но её процесс не найден (нет pid-файла "
                             "и порт не отвечает).\nОстановите вручную." % m.name)
            return
        self._stopping[m.path] = time.monotonic()
        threading.Thread(target=self._stop_worker, args=(m, pid), daemon=True).start()

    def _stop_worker(self, m, pid):
        _, pidfile = self._log_pid_files(m)
        self._log_event(m, "остановка")
        llm.stop_backend(pidfile, pid)
        if pid and llm.pid_alive(pid):
            # процесс в uninterruptible-IO (загрузка модели) — добиваем SIGKILL
            llm.ensure_stopped(pid)
            if os.path.exists(pidfile):
                try:
                    os.remove(pidfile)
                except OSError:
                    pass
        GLib.idle_add(self._stop_done, m)

    def _stop_done(self, m):
        self._stopping.pop(m.path, None)
        self.error_state = None
        self.tick()
        return False

    def _start_model(self, b, m):
        other = [x for x in self.models
                 if x.path in self.running and x.path != m.path]
        if other:
            names = ", ".join(x.name for x in other)
            if not self._confirm("Запуск",
                                 "Сейчас работает: %s.\nВместе модели не влезают в память — "
                                 "остановить и запустить «%s»?" % (names, m.name)):
                return
        self._starting[m.path] = time.monotonic()
        threading.Thread(target=self._start_worker, args=(b, m, other),
                         daemon=True).start()

    def _start_worker(self, b, m, other):
        # сначала остановить другие модели (синхронно, в этом потоке)
        for x in other:
            _, xpf = self._log_pid_files(x)
            xpid = llm.read_pid(xpf)
            if xpid:
                llm.stop_backend(xpf, xpid)
        cfg = self.cfg
        logfile, pidfile = self._log_pid_files(m)
        host, port = self._model_host_port(b, m)
        errs = [t for lvl, t in llm.preflight(b, m, cfg) if lvl == "error"]
        if errs:
            GLib.idle_add(self._start_failed, m, "; ".join(errs))
            return
        self._log_event(m, "запуск «%s» (порт %d)" % (m.name, port))
        pid, err = llm.start_backend(b, m, cfg, logfile, pidfile)
        if err:
            GLib.idle_add(self._start_failed, m, err)
            return
        ok, werr = llm.wait_ready(host, port, pid, timeout_s=1800)
        if ok:
            GLib.idle_add(self._start_ok, m)
        elif m.path in self._cancelled:
            GLib.idle_add(self._start_cancelled, m)
        else:
            GLib.idle_add(self._start_failed, m, werr)

    def _start_ok(self, m):
        self._starting.pop(m.path, None)
        self._cancelled.discard(m.path)
        self.error_state = None
        self.tick()
        return False

    def _start_cancelled(self, m):
        self._starting.pop(m.path, None)
        self._cancelled.discard(m.path)
        self.tick()
        return False

    def _start_failed(self, m, err):
        self._starting.pop(m.path, None)
        self._cancelled.discard(m.path)
        self.error_state = err
        self._log_event(m, "СБОЙ: %s" % err)
        logfile, _ = self._log_pid_files(m)
        tail = _tail(logfile, 20)
        body = "Не удалось запустить «%s»:\n\n%s" % (m.name, err)
        if tail.strip():
            body += "\n\nПоследние строки лога (%s):\n%s" % (logfile, tail.strip())
        self._dialog("Ошибка запуска", body)
        self.tick()
        return False

    def _on_stop_service(self):
        # жёсткая остановка всего (запущенного и загружающегося); апплет остаётся в трее
        print("[%s] Остановка сервиса: running=%s starting=%s"
              % (time.strftime("%H:%M:%S"),
                 [self._name_of(p) for p in self.running],
                 [self._name_of(p) for p in self._starting]), flush=True)
        self._cancelled.update(self._starting.keys())
        self._starting.clear()
        for b in self.backends:
            for m in self.by_backend.get(b.id, []):
                host, port = self._model_host_port(b, m)
                _, pidfile = self._log_pid_files(m)
                pid = llm.read_pid(pidfile) or llm.find_pid_by_port(port)
                if pid:
                    self._stopping[m.path] = time.monotonic()
                    threading.Thread(target=self._stop_worker, args=(m, pid),
                                     daemon=True).start()

    def _on_quit(self):
        # «Выход»: остановить и выгрузить все запущенные модели, затем завершить апплет
        for b in self.backends:
            for m in self.by_backend.get(b.id, []):
                host, port = self._model_host_port(b, m)
                _, pidfile = self._log_pid_files(m)
                pid = llm.read_pid(pidfile) or llm.find_pid_by_port(port)
                if pid:
                    llm.stop_backend(pidfile, pid)
                    if llm.pid_alive(pid):
                        llm.ensure_stopped(pid, timeout_s=60)
        Gtk.main_quit()

    # ------------------------------------------------------------- окна
    def _all_windows(self):
        return [w for w in (self.win_settings, self.win_logs, self.win_preflight)
                if w is not None]

    def _present_window(self, win):
        """Показать одно окно поверх, закрыв остальные (псевдо-модальность)."""
        for w in self._all_windows():
            if w is not win:
                w.hide()
        win.set_keep_above(True)
        if not win.get_visible():
            win.show_all()
        win.present()

    def _dismiss_window(self, win):
        win.set_keep_above(False)
        win.hide()

    def _on_focus_out(self, win, event):
        # клик вне окна (или по иконке трея) — убрать окно
        self._dismiss_window(win)
        return False

    def _on_delete(self, win, event):
        self._dismiss_window(win)
        return True

    def _place_top_right(self, win):
        d = Gdk.Display.get_default()
        mon = None
        try:
            w = win.get_window()
            if w:
                mon = d.get_monitor_at_window(w)
        except Exception:
            pass
        if mon is None:
            mon = d.get_monitor(0)
        if mon is None:
            return
        wa = mon.get_workarea()
        ww, wh = win.get_size()
        win.move(wa.x + wa.width - ww - 16, wa.y + 16)

    def _on_map_place(self, win, event):
        self._place_top_right(win)
        return False

    # -- настройки ---------------------------------------------------------
    def build_settings_window(self):
        self.win_settings = Gtk.Window(title="Стартовая настройка — servstart")
        self.win_settings.set_default_size(440, 560)
        self.win_settings.set_size_request(400, 380)
        self.win_settings.connect("delete-event", self._on_delete)
        self.win_settings.connect("focus-out-event", self._on_focus_out)
        self.win_settings.connect("map-event", self._on_map_place)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        root.set_border_width(10)

        head = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        lbl = Gtk.Label(label="Бэкенд:")
        self.backend_combo = Gtk.ComboBoxText()
        wired = [b for b in self.backends if b.wired]
        for b in wired:
            self.backend_combo.append(b.id, b.name)
        if wired:
            self.backend_combo.set_active(0)
        self.backend_combo.connect("changed", self._on_backend_changed)
        head.pack_start(lbl, False, False, 0)
        head.pack_start(self.backend_combo, True, True, 0)
        root.pack_start(head, False, False, 0)

        note = Gtk.Label(label="Настройки передаются бэкенду аргументами при запуске. "
                               "Применяются только по кнопке «ОК».")
        note.set_line_wrap(True)
        note.get_style_context().add_class("hint")
        root.pack_start(note, False, False, 0)

        self.settings_scroll = Gtk.ScrolledWindow()
        self.settings_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        self.settings_scroll.add(self.settings_box)
        root.pack_start(self.settings_scroll, True, True, 0)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        rst = Gtk.Button.new_with_label("Сброс")
        rst.connect("clicked", self._on_reset_settings)
        ok = Gtk.Button.new_with_label("ОК")
        ok.get_style_context().add_class("suggested-action")
        ok.connect("clicked", self._on_save_settings)
        cancel = Gtk.Button.new_with_label("Отмена")
        cancel.connect("clicked", lambda *a: self._dismiss_window(self.win_settings))
        btns.pack_start(rst, False, False, 0)
        btns.pack_end(cancel, False, False, 0)
        btns.pack_end(ok, False, False, 0)
        root.pack_start(btns, False, False, 0)

        self.win_settings.add(root)

    def _current_backend(self):
        bid = self.backend_combo.get_active_id()
        return next((b for b in self.backends if b.id == bid), None)

    def _on_backend_changed(self, combo):
        self._rebuild_settings_pane()

    def _rebuild_settings_pane(self):
        for c in self.settings_box.get_children():
            self.settings_box.remove(c)
        self._widgets = {}
        b = self._current_backend()
        if not b:
            return
        for s in b.settings:
            self.settings_box.pack_start(self._setting_row(s), False, False, 0)
        self.settings_box.show_all()

    def _setting_row(self, s):
        row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=3)
        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)

        lab = Gtk.Label(label=s.label)
        lab.set_xalign(0.0)
        lab.get_style_context().add_class("title")
        top.pack_start(lab, False, False, 0)

        val = (self.cfg.get("backends") or {}).get(
            self._current_backend().id, {}).get(s.key, s.default)

        expand = False
        if s.type == "int":
            lo = s.min if s.min is not None else 0
            hi = s.max if s.max is not None else 2 ** 31 - 1
            w = Gtk.SpinButton.new_with_range(lo, hi, s.step)
            w.set_value(int(val or 0))
        elif s.type == "bool":
            w = Gtk.Switch()
            w.set_active(bool(val))
        elif s.type == "enum":
            w = Gtk.ComboBoxText()
            for c in s.choices or []:
                w.append(c, c)
            w.set_active_id(str(val))
        else:  # str — текстовое поле разумно растянуть
            w = Gtk.Entry()
            w.set_text("" if val is None else str(val))
            expand = True
        w.set_valign(Gtk.Align.CENTER)
        self._widgets[s.key] = w
        top.pack_end(w, expand, expand, 0)

        row.pack_start(top, False, False, 0)

        hint = Gtk.Label(label="что: %s\nдля чего: %s" % (s.what, s.why))
        hint.set_xalign(0.0)
        hint.set_line_wrap(True)
        hint.get_style_context().add_class("hint")
        row.pack_start(hint, False, False, 0)
        return row

    def _read_settings(self):
        b = self._current_backend()
        out = {}
        for s in b.settings:
            w = self._widgets[s.key]
            if s.type == "int":
                out[s.key] = w.get_value_as_int()
            elif s.type == "bool":
                out[s.key] = w.get_active()
            elif s.type == "enum":
                out[s.key] = w.get_active_id()
            else:
                out[s.key] = w.get_text()
        return out

    def _on_save_settings(self, btn):
        b = self._current_backend()
        if not b:
            return
        self.cfg.setdefault("backends", {})
        self.cfg["backends"][b.id] = self._read_settings()
        llm.save_config(self.cfg)
        self._dismiss_window(self.win_settings)
        self.tick()

    def _on_reset_settings(self, btn):
        b = self._current_backend()
        if not b:
            return
        self.cfg.setdefault("backends", {})
        self.cfg["backends"].pop(b.id, None)
        llm.save_config(self.cfg)
        self._rebuild_settings_pane()

    def show_settings(self):
        self._rebuild_settings_pane()   # сброс к сохранённому конфигу
        self._present_window(self.win_settings)

    # -- логи ---------------------------------------------------------------
    def build_logs_window(self):
        self.win_logs = Gtk.Window(title="Логи — servstart")
        self.win_logs.set_default_size(720, 480)
        self.win_logs.connect("delete-event", self._on_delete)
        self.win_logs.connect("focus-out-event", self._on_focus_out)
        self.win_logs.connect("map-event", self._on_map_place)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        root.set_border_width(10)

        top = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        lbl = Gtk.Label(label="Лог:")
        self.log_combo = Gtk.ComboBoxText()
        self.log_paths = {}
        for b in self.backends:
            for m in self.by_backend.get(b.id, []):
                logfile, _ = self._log_pid_files(m)
                self.log_combo.append(m.path, "%s · %s" % (b.name, m.name))
                self.log_paths[m.path] = logfile
        if self.log_paths:
            self.log_combo.set_active(0)
        self.log_combo.connect("changed", self._on_log_changed)
        top.pack_start(lbl, False, False, 0)
        top.pack_start(self.log_combo, True, True, 0)
        root.pack_start(top, False, False, 0)

        sc = Gtk.ScrolledWindow()
        sc.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_cursor_visible(False)
        self.log_view.set_monospace(True)
        self.log_view.set_wrap_mode(Gtk.WrapMode.NONE)   # строки не переносить
        sc.add(self.log_view)
        root.pack_start(sc, True, True, 0)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        refresh = Gtk.Button.new_with_label("Обновить")
        refresh.connect("clicked", lambda *a: self._show_log_session())
        term = Gtk.Button.new_with_label("В терминале (tail -f)")
        term.connect("clicked", self._open_log_in_terminal)
        folder = Gtk.Button.new_with_label("Показать файл")
        folder.connect("clicked", self._open_log_file)
        close = Gtk.Button.new_with_label("Закрыть")
        close.connect("clicked", lambda *a: self._dismiss_window(self.win_logs))
        btns.pack_start(refresh, False, False, 0)
        btns.pack_start(close, False, False, 0)
        btns.pack_end(folder, False, False, 0)
        btns.pack_end(term, False, False, 0)
        root.pack_start(btns, False, False, 0)

        self.win_logs.add(root)
        GLib.timeout_add_seconds(2, self._log_tick)

    def _current_log_path(self):
        mid = self.log_combo.get_active_id()
        return self.log_paths.get(mid)

    def _on_log_changed(self, combo):
        self._show_log_session()

    def _log_tick(self):
        if self.win_logs.get_visible():
            self._refresh_log()
        return True

    def _stamp(self, line, now):
        # строки-маркеры servstart уже несут свой timestamp — не дублируем
        if line.startswith("[20") and "servstart:" in line:
            return line
        return "[%s] %s" % (now, line)

    def _log_marker_pos(self, logfile):
        """Смещение начала последнего сеанса (последний маркер 'servstart: запуск')."""
        try:
            with open(logfile, "r", errors="replace") as f:
                data = f.read()
        except OSError:
            return None
        idx = data.rfind("servstart: запуск")
        if idx == -1:
            idx = data.rfind("servstart: СБОЙ")
        if idx == -1:
            return None
        return data.rfind("\n", 0, idx) + 1

    def _read_log_increment(self, logfile):
        """Читает только новые байты с прошлой позиции. Возвращает список новых строк."""
        try:
            st = os.stat(logfile)
        except OSError:
            return []
        state = self._log_state.setdefault(
            logfile, {"pos": 0, "inode": None, "partial": ""})
        if state["inode"] != st.st_ino:
            state.update(pos=0, inode=st.st_ino, partial="")
        pos = state["pos"]
        if st.st_size < pos:
            pos = 0  # файл обрезан
        with open(logfile, "r", errors="replace") as f:
            f.seek(pos)
            data = f.read()
            state["pos"] = f.tell()
        state["inode"] = st.st_ino
        if not data:
            return []
        buf = state["partial"] + data
        lines = buf.split("\n")
        state["partial"] = lines[-1]
        return lines[:-1]

    def _scroll_log_to_end(self):
        buf = self.log_view.get_buffer()
        end = buf.get_end_iter()
        self.log_view.scroll_to_iter(end, 0.0, False, 0.0, 0.0)
        return False

    def _show_log_session(self):
        """Показать текущий сеанс (с последнего маркера запуска), не весь файл."""
        p = self._current_log_path()
        if not p:
            return
        buf = self.log_view.get_buffer()
        now = time.strftime("%H:%M:%S")
        try:
            st = os.stat(p)
        except OSError:
            buf.set_text("[%s] (лога нет: %s)" % (now, p))
            self._log_state.pop(p, None)
            return
        marker = self._log_marker_pos(p)
        lines = []
        with open(p, "r", errors="replace") as f:
            if marker is not None:
                f.seek(marker)
                lines = f.read().splitlines()
            else:
                lines = _tail(p, 40).splitlines()
        # дальше читаем только новое (не перегружаем весь файл)
        self._log_state[p] = {"pos": st.st_size, "inode": st.st_ino, "partial": ""}
        text = "\n".join(self._stamp(l, now) for l in lines)
        buf.set_text(text or "[%s] — сеанс начат, ждём события…" % now)
        self._fit_window_width(self.win_logs, self.log_view, text)
        GLib.idle_add(self._scroll_log_to_end)

    def _append_log_lines(self, lines):
        """Дописать новые строки вниз с [HH:MM:SS] и прокрутить (новое снизу)."""
        if not lines:
            return
        now = time.strftime("%H:%M:%S")
        buf = self.log_view.get_buffer()
        end = buf.get_end_iter()
        text = "".join(self._stamp(l, now) + "\n" for l in lines)
        buf.insert(end, text)
        # держим в буфере только последние ~400 строк
        n = buf.get_line_count()
        if n > 400:
            start = buf.get_iter_at_line(n - 400)
            buf.delete(buf.get_start_iter(), start)
        GLib.idle_add(self._scroll_log_to_end)

    def _refresh_log(self):
        p = self._current_log_path()
        if not p:
            return
        self._append_log_lines(self._read_log_increment(p))

    def _fit_window_width(self, win, view, text, min_w=520):
        """Расширить окно по ширине самой длинной строки (уместить без обрезки)."""
        if not win.get_visible():
            return
        maxlen = max((len(l) for l in text.splitlines()), default=0)
        layout = view.create_pango_layout("M" * min(maxlen, 500))
        w, _ = layout.get_pixel_size()
        target = w + 80
        screen = Gdk.Screen.get_default()
        avail = screen.get_width() if screen else 1920
        target = max(min_w, min(target, int(avail * 0.92)))
        cur = win.get_size()[0]
        if abs(target - cur) > 10:
            win.resize(target, win.get_size()[1])
            self._place_top_right(win)

    def _open_log_in_terminal(self, btn):
        p = self._current_log_path()
        if not p or not os.path.isfile(p):
            return
        for term in ("x-terminal-emulator", "gnome-terminal", "kgx", "konsole", "xterm"):
            if not subprocess.call(["which", term], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL):
                subprocess.Popen([term, "-e", "tail -f %s" % p],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        self._dialog("Лог", "Не найден терминал. Файл лога:\n%s" % p)

    def _open_log_file(self, btn):
        p = self._current_log_path()
        if p and os.path.isfile(p):
            subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

    def show_logs(self):
        # выбрать запущенную модель, иначе — последнюю по свежести лога
        sel = None
        running_paths = sorted(self.running)
        if running_paths:
            sel = running_paths[0]
        else:
            best, best_mt = None, 0.0
            for path, logfile in self.log_paths.items():
                try:
                    mt = os.path.getmtime(logfile)
                except OSError:
                    mt = 0.0
                if mt > best_mt:
                    best_mt, best = mt, path
            sel = best or next(iter(self.log_paths), None)
        if sel and sel in self.log_paths:
            self.log_combo.set_active_id(sel)
        self._present_window(self.win_logs)
        self._show_log_session()

    # -- готовность ---------------------------------------------------------
    def build_preflight_window(self):
        self.win_preflight = Gtk.Window(title="Готовность — servstart")
        self.win_preflight.set_default_size(560, 460)
        self.win_preflight.connect("delete-event", self._on_delete)
        self.win_preflight.connect("focus-out-event", self._on_focus_out)
        self.win_preflight.connect("map-event", self._on_map_place)

        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        root.set_border_width(10)

        sc = Gtk.ScrolledWindow()
        sc.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.preflight_view = Gtk.TextView()
        self.preflight_view.set_editable(False)
        self.preflight_view.set_cursor_visible(False)
        self.preflight_view.set_monospace(True)
        self.preflight_view.set_wrap_mode(Gtk.WrapMode.NONE)
        sc.add(self.preflight_view)
        root.pack_start(sc, True, True, 0)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        refresh = Gtk.Button.new_with_label("Обновить")
        refresh.connect("clicked", lambda *a: self.show_preflight())
        close = Gtk.Button.new_with_label("Закрыть")
        close.connect("clicked", lambda *a: self._dismiss_window(self.win_preflight))
        btns.pack_end(close, False, False, 0)
        btns.pack_end(refresh, False, False, 0)
        root.pack_start(btns, False, False, 0)

        self.win_preflight.add(root)

    def show_preflight(self):
        lines = []
        for b in self.backends:
            for m in self.by_backend.get(b.id, []):
                lines.append("— %s · %s —" % (b.name, m.name))
                for lvl, t in llm.preflight(b, m, self.cfg):
                    mark = {"ok": "✓", "warn": "!", "error": "✗"}[lvl]
                    lines.append("  %s %s" % (mark, t))
        text = "\n".join(lines) or "бэкенды не найдены"
        self.preflight_view.get_buffer().set_text(text)
        self._present_window(self.win_preflight)
        self._fit_window_width(self.win_preflight, self.preflight_view, text)

    # ------------------------------------------------------------- диалоги
    def _dialog(self, title, body):
        d = Gtk.MessageDialog(transient_for=None, modal=True,
                              message_type=Gtk.MessageType.ERROR,
                              buttons=Gtk.ButtonsType.OK)
        d.set_title(title)
        d.props.text = body
        d.run()
        d.destroy()

    def _confirm(self, title, body):
        d = Gtk.MessageDialog(transient_for=None, modal=True,
                              message_type=Gtk.MessageType.QUESTION,
                              buttons=Gtk.ButtonsType.YES_NO)
        d.set_title(title)
        d.props.text = body
        r = d.run()
        d.destroy()
        return r == Gtk.ResponseType.YES


def main():
    ServStart()
    Gtk.main()


if __name__ == "__main__":
    main()
