"""Interactive sandbox for Lucía's attention system."""

from .attention import AttentionEngine
from .events import Event


def ask_score(label: str) -> float:
    while True:
        raw = input(f"{label} [0-1]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Introduce un número, por ejemplo 0.7")
            continue
        if 0.0 <= value <= 1.0:
            return value
        print("El valor debe estar entre 0 y 1.")


def main() -> None:
    attention = AttentionEngine()
    print("Lucía — Attention Sandbox")
    print("Escribe un evento y puntúa sus propiedades. Ctrl+C para salir.\n")

    try:
        while True:
            text = input("Evento > ").strip()
            if not text:
                continue

            event = Event(type="user.observation", data={"text": text}, source="sandbox")
            novelty = ask_score("Novedad")
            relevance = ask_score("Relevancia")
            urgency = ask_score("Urgencia")
            goal_alignment = ask_score("Alineación con objetivo")

            score = attention.score(
                event,
                novelty=novelty,
                relevance=relevance,
                urgency=urgency,
                goal_alignment=goal_alignment,
            )
            decision = attention.should_process(score)

            print(f"\nSalience: {score:.2f}")
            print("DECISIÓN: PRESTAR ATENCIÓN" if decision else "DECISIÓN: IGNORAR POR AHORA")
            print("-" * 40)
    except KeyboardInterrupt:
        print("\nLucía sandbox cerrado.")


if __name__ == "__main__":
    main()
