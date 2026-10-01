#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор иконок-«светофора» для servstart (чистый Python, без PIL/cairo).

Рисует круглый индикатор с тёмной обводкой и мягким антиалиасингом
(SDF круга), сохраняет RGBA-PNG. Используется и в install.sh, и на лету,
чтобы иконка никогда не была бинарным блобом в репозитории.

    python3 gen_icons.py [каталог]   # по умолчанию ./icons
"""
import os
import struct
import sys
import zlib

SIZE = 64                     # px
BORDER = (0x2b, 0x2b, 0x2b)   # тёмная обводка (r, g, b)


def _chunk(tag, data):
    c = struct.pack(">I", len(data)) + tag + data
    c += struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return c


def make_icon(path, fill_rgb, size=SIZE, border=BORDER):
    """fill_rgb — кортеж (r, g, b)."""
    cx = cy = (size - 1) / 2.0
    radius = size * 0.40          # радиус заливки
    border_r = size * 0.46        # внешний радиус ободка
    rows = []
    for y in range(size):
        row = bytearray([0])      # фильтр 0 (None) на строку
        for x in range(size):
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            # заливка (мягкий край)
            fill_a = max(0.0, min(1.0, radius - d + 0.5))
            # ободок: кольцо между fill и border_r
            ring = max(0.0, min(1.0, border_r - d + 0.5))
            # пиксели за внешним краем — прозрачны
            if ring <= 0.0:
                row += b"\x00\x00\x00\x00"
                continue
            # смешиваем: сначала кольцо (border), поверх — заливка
            r, g, b = border
            a = ring
            if fill_a > 0.0:
                r, g, b = fill_rgb
                a = fill_a
            # краевые пиксели кольца — приглушаем прозрачность до ring
            if ring < 1.0 and fill_a <= 0.0:
                a = ring
            row += bytes((r, g, b, int(round(a * 255))))
        rows.append(bytes(row))

    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", zlib.compress(raw, 9))
           + _chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "icons")
    os.makedirs(out_dir, exist_ok=True)
    specs = {
        "servstart-gray.png": (0x88, 0x88, 0x88),   # ничего не запущено
        "servstart-green.png": (0x30, 0xc5, 0x3a),  # запущено
        "servstart-red.png": (0xe0, 0x3a, 0x3a),    # сбой / ошибка
    }
    for name, rgb in specs.items():
        make_icon(os.path.join(out_dir, name), rgb)
        print("ok", os.path.join(out_dir, name))


if __name__ == "__main__":
    main()
