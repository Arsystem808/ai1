from __future__ import annotations
import random

def _verbal_conf(c: float) -> str:
    if c < 0.45: return "низкая"
    if c < 0.7:  return "средняя"
    return "высокая"

def _dir(action: str) -> str:
    return {"BUY":"покупку","SHORT":"шорт","WAIT":"ожидание"}.get(action,"ожидание")

def _nearest(sig: dict):
    px=float(sig["entry"])
    ups=[sig.get("R1"),sig.get("R2"),sig.get("R3"),sig.get("upper_zone")]
    dns=[sig.get("S1"),sig.get("S2"),sig.get("S3"),sig.get("lower_zone")]
    ups=sorted([float(x) for x in ups if isinstance(x,(int,float)) and x>px])[:2]
    dns=sorted([float(x) for x in dns if isinstance(x,(int,float)) and x<px], reverse=True)[:2]
    return ups,dns

def build_rationale(symbol: str, horizon_ui: str, sig: dict, detail: str="Стандарт") -> str:
    action=sig.get("action","WAIT"); conf=_verbal_conf(float(sig.get("confidence",0.5)))
    ups,dns=_nearest(sig)
    intro = f"{symbol}: данные приняты, режим — {horizon_ui.lower()}."
    plan  = f"План: вход {sig['entry']}, стоп {sig['sl']}, цели {sig['tp1']} / {sig['tp2']}. Ключевая {sig['key_mark']}."
    ctx   = []
    if ups: ctx.append("Сверху ориентиры: " + ", ".join(f\"{u:.2f}\" for u in ups) + ".")
    if dns: ctx.append("Снизу ориентиры: " + ", ".join(f\"{d:.2f}\" for d in dns) + ".")
    tail  = "Ждём реакции у ближайших отметок; уверенный выход задаст ход."
    short = f"Сценарий: {_dir(action)}. Уверенность — {conf}."
    if detail=="Коротко": parts=[short, plan, tail]
    elif detail=="Подробно":
        nuance=random.choice([
            "Приоритет — работа от уровня и контроль риска.",
            "Импульс подтверждаем по реакции на пробой/отбой.",
            "Часть позиции фиксируем у первой цели."
        ])
        parts=[intro, short, plan, " ".join(ctx), nuance, tail]
    else:
        parts=[intro, short, plan, " ".join(ctx), tail]
    return " ".join([p for p in parts if p]).replace("  "," ").strip()

