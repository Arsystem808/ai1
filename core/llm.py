from __future__ import annotations
import random

def _verbal_conf(c: float) -> str:
    if c < 0.45: return "низкая"
    if c < 0.7:  return "средняя"
    return "высокая"

def _dir_word(action: str) -> str:
    return {"BUY":"покупку","SHORT":"шорт","WAIT":"ожидание","CLOSE":"закрытие"}.get(action, "ожидание")

def _nearest_levels(sig: dict) -> tuple[list[float], list[float]]:
    px = float(sig.get("entry", sig.get("Close", 0.0)))
    up_raw = [sig.get("upper_zone"), sig.get("key_mark"), sig.get("R1"), sig.get("R2"), sig.get("R3")]
    dn_raw = [sig.get("lower_zone"), sig.get("key_mark"), sig.get("S1"), sig.get("S2"), sig.get("S3")]
    ups = sorted([float(x) for x in up_raw if isinstance(x,(int,float)) and x and x > px])[:2]
    dns = sorted([float(x) for x in dn_raw if isinstance(x,(int,float)) and x and x < px], reverse=True)[:2]
    return ups, dns

def _intro(symbol: str, action: str, src: str, horizon_ui: str) -> str:
    starts = [
        f"{symbol}: картина спокойная. Источник — {src.lower()}, горизонт — {horizon_ui.lower()}.",
        f"{symbol}: рынок ведёт себя ровно; данные — {src.lower()}, горизонт — {horizon_ui.lower()}.",
        f"{symbol}: тон умеренный. Берём {src.lower()}, режим — {horizon_ui.lower()}."
    ]
    if action in ("BUY","SHORT"):
        starts += [
            f"{symbol}: условия для сделки выглядят рабочими. Источник — {src}, горизонт — {horizon_ui.lower()}.",
            f"{symbol}: есть сетап под {_dir_word(action)}. Котировки {src.lower()}, горизонт — {horizon_ui.lower()}."
        ]
    return random.choice(starts)

def _plan(sig: dict) -> str:
    return (f"План: вход {sig['entry']}, стоп {sig['sl']}, цели {sig['tp1']} / {sig['tp2']}. "
            f"Рабочий коридор {sig['lower_zone']}–{sig['upper_zone']}, опорная отметка {sig['key_mark']}.")

def _context_levels(sig: dict) -> str:
    ups, dns = _nearest_levels(sig)
    parts = []
    if ups:
        parts.append(("Сверху ближайшие ориентиры — " + ", ".join(f"{u:.2f}" for u in ups)) + ".")
    if dns:
        parts.append(("Снизу ориентиры — " + ", ".join(f"{d:.2f}" for d in dns)) + ".")
    return " ".join(parts)

def _next_steps(action: str) -> str:
    if action == "BUY":
        return "Если цена удержится выше ближайшего сопротивления, даём ходу; при вялой реакции — сокращаем риск."
    if action == "SHORT":
        return "Если давление сохранится у ближайшей поддержки, держим ход до второй цели; при отскоке — защищаемся."
    if action == "CLOSE":
        return "Фиксируем результат и смотрим за новой расстановкой сил."
    return "Наблюдаем за реакцией у ближайших ориентиров; уверенный пробой задаст следующее движение."

def build_rationale(symbol: str, horizon_ui: str, sig: dict, detail: str = "Стандарт") -> str:
    action = sig.get("action", "WAIT")
    source = sig.get("source", "market")
    conf_t = _verbal_conf(float(sig.get("confidence", 0.5)))

    intro = _intro(symbol, action, source, horizon_ui)
    plan  = _plan(sig)
    lvl   = _context_levels(sig)
    tail  = _next_steps(action)
    conf  = f"Уверенность — {conf_t}."

    if detail == "Коротко":
        parts = [f"Сценарий: {_dir_word(action)}.", conf, plan, tail]
    elif detail == "Подробно":
        nuances = [
            "Следим за поведением в узком коридоре: импульс часто появляется на выходе.",
            "Риски контролируем через стоп и частичную фиксацию у первой цели.",
            "Общий фон учитываем, но решения привязываем к своим уровням."
        ]
        parts = [intro, f"Сценарий: {_dir_word(action)}.", conf, plan, lvl, random.choice(nuances), tail]
    else:
        parts = [intro, f"Сценарий: {_dir_word(action)}.", conf, plan, lvl, tail]

    return " ".join([p for p in parts if p]).replace("  ", " ").strip()
