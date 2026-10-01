#!/usr/bin/env bash
# Установка servstart: трей-сервис управления LLM-бэкендами и моделями.
# Полностью user-scope, sudo НЕ требуется (бинарники и модели — пользовательские).
# Запускать из папки пакета: servstart.py, llm.py, backends.json, gen_icons.py,
# servstart.service.
set -euo pipefail
cd "$(dirname "$0")"

DEST="$HOME/.local/lib/servstart"
LAUNCHER="$HOME/.local/bin/servstart"
UNIT_DST="$HOME/.config/systemd/user/servstart.service"
PY=/usr/bin/python3

echo "==> Проверка интерпретатора ($PY, PyGObject)"
if ! "$PY" -c 'import gi; gi.require_version("Gtk","3.0"); from gi.repository import Gtk' 2>/dev/null; then
    echo "ОШИБКА: $PY не видит PyGObject. Установите:"
    echo "  sudo apt-get install -y python3-gi gir1.2-gtk-3.0 gir1.2-ayatanaappindicator3-0.1"
    exit 1
fi
if ! "$PY" -c 'import gi; gi.require_version("AyatanaAppIndicator3","0.1")' 2>/dev/null; then
    echo "ПРЕДУПРЕЖДЕНИЕ: нет AyatanaAppIndicator3 — апплет запустится без трей-иконки."
fi

echo "==> Генерация иконок-светофора"
"$PY" gen_icons.py icons

echo "==> Копирование файлов в $DEST"
mkdir -p "$DEST/icons"
install -m755 servstart.py  "$DEST/servstart.py"
install -m644 llm.py       "$DEST/llm.py"
install -m644 backends.json "$DEST/backends.json"
install -m644 icons/servstart-gray.png  "$DEST/icons/"
install -m644 icons/servstart-green.png "$DEST/icons/"
install -m644 icons/servstart-red.png   "$DEST/icons/"

echo "==> Лаунчер $LAUNCHER"
ln -sf "$DEST/servstart.py" "$LAUNCHER"

echo "==> systemd user-юнит"
install -m644 servstart.service "$UNIT_DST"
systemctl --user daemon-reload
systemctl --user enable --now servstart.service

echo
echo "==> Готово. Автозапуск:"
systemctl --user is-enabled servstart.service
systemctl --user is-active  servstart.service
echo
echo "Логи апплета:  journalctl --user -u servstart -f"
echo "Остановить:    systemctl --user stop servstart"
echo "Удалить:       systemctl --user disable --now servstart.service; rm -rf $DEST $LAUNCHER $UNIT_DST; systemctl --user daemon-reload"
