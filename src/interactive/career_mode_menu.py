
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import time
import pandas as pd
import numpy as np
from IPython.display import display, HTML, clear_output

try:
    import ipywidgets as widgets
except Exception:
    widgets = None


TEAM_FLAGS = {
    "Mexico": "🇲🇽",
    "South Africa": "🇿🇦",
    "South Korea": "🇰🇷",
    "Czechia": "🇨🇿",
    "Canada": "🇨🇦",
    "Bosnia and Herzegovina": "🇧🇦",
    "Qatar": "🇶🇦",
    "Switzerland": "🇨🇭",
    "Brazil": "🇧🇷",
    "Morocco": "🇲🇦",
    "Haiti": "🇭🇹",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
    "United States": "🇺🇸",
    "Paraguay": "🇵🇾",
    "Australia": "🇦🇺",
    "Turkey": "🇹🇷",
    "Germany": "🇩🇪",
    "Curacao": "🇨🇼",
    "Ivory Coast": "🇨🇮",
    "Ecuador": "🇪🇨",
    "Netherlands": "🇳🇱",
    "Japan": "🇯🇵",
    "Sweden": "🇸🇪",
    "Tunisia": "🇹🇳",
    "Belgium": "🇧🇪",
    "Egypt": "🇪🇬",
    "Iran": "🇮🇷",
    "New Zealand": "🇳🇿",
    "Spain": "🇪🇸",
    "Cape Verde": "🇨🇻",
    "Saudi Arabia": "🇸🇦",
    "Uruguay": "🇺🇾",
    "France": "🇫🇷",
    "Senegal": "🇸🇳",
    "Iraq": "🇮🇶",
    "Norway": "🇳🇴",
    "Argentina": "🇦🇷",
    "Algeria": "🇩🇿",
    "Austria": "🇦🇹",
    "Jordan": "🇯🇴",
    "Portugal": "🇵🇹",
    "DR Congo": "🇨🇩",
    "Uzbekistan": "🇺🇿",
    "Colombia": "🇨🇴",
    "England": "🏴",
    "Croatia": "🇭🇷",
    "Ghana": "🇬🇭",
    "Panama": "🇵🇦",
}


def flag(team: str) -> str:
    return TEAM_FLAGS.get(str(team), "🏳️")


def team_label(team: str) -> str:
    return f"{flag(team)} {team}"


def whtml(html: str):
    if widgets is not None:
        return widgets.HTML(value=html)
    return HTML(html)


def display_html(html: str):
    display(HTML(html))


def header_widget():
    return whtml(
        "<div style='padding:22px;border-radius:20px;"
        "background:linear-gradient(135deg,#5A002A,#071B3A 65%,#0B5D4A);"
        "color:white;margin-bottom:14px;border:1px solid rgba(255,255,255,.25);'>"
        "<div style='font-size:13px;letter-spacing:2px;color:#F5D06F;font-weight:700;'>MUNDIAL 2026 · MODO COPA</div>"
        "<h2 style='margin:6px 0 4px 0;'>🏆 Simulador tipo modo carrera</h2>"
        "<p style='margin:0;color:#E6E8EC;'>Simula grupos, construye la llave y avanza hasta conocer al campeón.</p>"
        "</div>"
    )


def note_widget(text: str):
    return whtml(
        f"<div style='padding:12px;border-radius:14px;background:#F8FAFC;"
        f"border:1px solid #EAECF0;color:#344054;margin:8px 0;'>{text}</div>"
    )


def ok_html(text: str):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#ECFDF3;"
        f"border:1px solid #ABEFC6;color:#027A48;margin:10px 0;'><b>{text}</b></div>"
    )


def warn_html(text: str):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#FFFAEB;"
        f"border:1px solid #FEDF89;color:#B54708;margin:10px 0;'><b>{text}</b></div>"
    )


def error_html(e):
    return (
        f"<div style='padding:14px;border-radius:14px;background:#FEF3F2;"
        f"border:1px solid #FECDCA;color:#B42318;margin:10px 0;'><b>Error:</b> {e}</div>"
    )


def render_r32_cards(r32_df: pd.DataFrame) -> str:
    cards = ""

    for _, row in r32_df.iterrows():
        allowed = row.get("allowed_thirds", "")
        allowed_line = ""
        if allowed:
            allowed_line = (
                f"<div style='font-size:11px;color:#667085;margin-top:6px;'>"
                f"Slot original: {allowed}</div>"
            )

        cards += f"""
        <div style='background:white;border:1px solid #EAECF0;border-radius:18px;
                    box-shadow:0 6px 16px rgba(16,24,40,.07);padding:14px;'>
            <div style='font-size:12px;color:#667085;font-weight:700;'>M{row["match_id"]} · {row["round"]}</div>
            <div style='display:flex;justify-content:space-between;align-items:center;margin-top:10px;'>
                <div style='font-size:15px;font-weight:800;color:#101828;'>{team_label(row["team_a"])}</div>
                <div style='font-size:12px;color:#98A2B3;'>({row["slot_a"]})</div>
            </div>
            <div style='text-align:center;color:#B42318;font-weight:900;margin:8px 0;'>VS</div>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div style='font-size:15px;font-weight:800;color:#101828;'>{team_label(row["team_b"])}</div>
                <div style='font-size:12px;color:#98A2B3;'>({row["slot_b_resolved"]})</div>
            </div>
            {allowed_line}
        </div>
        """

    return f"""
    <div style='margin:16px 0;'>
      <h3 style='margin-bottom:10px;color:#101828;'>Llave inicial · Ronda de 32</h3>
      <div style='display:grid;grid-template-columns:repeat(2,minmax(280px,1fr));gap:12px;'>
        {cards}
      </div>
    </div>
    """


def render_final_summary(champion: str, runner_up: str, third_place: str) -> str:
    return f"""
    <div style='padding:20px;border-radius:20px;
                background:linear-gradient(135deg,#5A002A,#071B3A);
                color:white;margin:16px 0;'>
      <div style='font-size:13px;color:#F5D06F;font-weight:700;letter-spacing:2px;'>RESULTADO FINAL</div>
      <h2 style='margin:8px 0;'>🏆 Campeón: {team_label(champion)}</h2>
      <p style='margin:4px 0;color:#E6E8EC;'><b>Subcampeón:</b> {team_label(runner_up)}</p>
      <p style='margin:4px 0;color:#E6E8EC;'><b>Tercer lugar:</b> {team_label(third_place)}</p>
    </div>
    """


def compact_group_tables_html(group_stage: dict) -> str:
    tables = group_stage["tables"]
    cards = ""

    for group, table in tables.items():
        t = table.sort_values("Pos").copy()
        rows = ""
        for _, r in t.iterrows():
            rows += (
                f"<tr>"
                f"<td style='padding:4px 6px;font-weight:700;'>{int(r['Pos'])}</td>"
                f"<td style='padding:4px 6px;'>{team_label(r['Equipo'])}</td>"
                f"<td style='padding:4px 6px;text-align:center;font-weight:800;'>{int(r['Pts'])}</td>"
                f"<td style='padding:4px 6px;text-align:center;'>{int(r['DG'])}</td>"
                f"</tr>"
            )

        cards += f"""
        <div style='background:white;border:1px solid #EAECF0;border-radius:16px;padding:10px;'>
          <div style='font-weight:900;color:#5A002A;margin-bottom:6px;'>Grupo {group}</div>
          <table style='width:100%;font-size:12px;border-collapse:collapse;'>
            <thead>
              <tr style='color:#667085;'>
                <th style='text-align:left;'>#</th>
                <th style='text-align:left;'>Equipo</th>
                <th>Pts</th>
                <th>DG</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """

    return f"""
    <div style='margin:16px 0;'>
      <h3 style='color:#101828;'>Tablas de grupos simuladas</h3>
      <div style='display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:12px;'>
        {cards}
      </div>
    </div>
    """


def display_bracket_table(bracket: pd.DataFrame):
    cols = [
        "match_id",
        "round",
        "home_team",
        "away_team",
        "home_goals_90",
        "away_goals_90",
        "home_goals_et",
        "away_goals_et",
        "winner",
        "method",
    ]
    cols = [c for c in cols if c in bracket.columns]
    display(bracket[cols])


def build_group_stage_with_progress(
    project_root: Path,
    analysis_date: str,
    use_registered_results: bool,
    seed: int,
    progress,
    status,
):
    from src.interactive.group_match_selector import GROUPS
    from src.simulation.group_stage_simulator import simulate_single_group, PredictionCache, rank_best_thirds

    rng = np.random.default_rng(seed)
    cache = PredictionCache()

    all_tables = {}
    all_results = []

    groups_sorted = sorted(GROUPS.keys())
    n_groups = len(groups_sorted)

    for i, group in enumerate(groups_sorted):
        pct = int(5 + (i / max(n_groups, 1)) * 35)
        progress.value = pct
        status.value = f"<b>[1/4]</b> Simulando grupo {group}..."

        table, results = simulate_single_group(
            group=group,
            project_root=project_root,
            analysis_date=analysis_date,
            use_registered_results=use_registered_results,
            rng=rng,
            prediction_cache=cache,
            corners_cards_mode="legacy",
        )

        all_tables[group] = table
        all_results.append(results)

    progress.value = 42
    status.value = "<b>[1/4]</b> Calculando mejores terceros..."

    best_thirds = rank_best_thirds(all_tables)

    qualifiers = {}
    for group, table in all_tables.items():
        ordered = table.sort_values("Pos").reset_index(drop=True)
        qualifiers[f"1{group}"] = ordered.iloc[0]["Equipo"]
        qualifiers[f"2{group}"] = ordered.iloc[1]["Equipo"]

    qualified_thirds = best_thirds[best_thirds["qualifies_as_best_third"]]
    for _, row in qualified_thirds.iterrows():
        qualifiers[f"3{row['Grupo']}"] = row["Equipo"]

    results_df = pd.concat(all_results, ignore_index=True)

    return {
        "tables": all_tables,
        "results": results_df,
        "best_thirds": best_thirds,
        "qualifiers": qualifiers,
    }


def build_r32_bracket_with_progress(group_stage: dict, progress, status):
    from src.simulation.tournament_simulator import R32_SLOTS, assign_third_place_slots, resolve_slot

    progress.value = 46
    status.value = "<b>[2/4]</b> Construyendo llave de ronda de 32..."

    qualifiers = group_stage["qualifiers"]
    best_thirds = group_stage["best_thirds"]

    qualified_third_groups = list(
        best_thirds[best_thirds["qualifies_as_best_third"]]["Grupo"]
    )

    third_assignment = assign_third_place_slots(qualified_third_groups)

    rows = []
    for slot in R32_SLOTS:
        slot_a = slot["slot_a"]
        slot_b_original = slot["slot_b"]

        if slot_b_original == "3?":
            slot_b_resolved = third_assignment[slot["match_id"]]
        else:
            slot_b_resolved = slot_b_original

        team_a = resolve_slot(slot_a, qualifiers)
        team_b = resolve_slot(slot_b_resolved, qualifiers)

        allowed = ""
        if slot_b_original == "3?":
            allowed = " / ".join(["3" + g for g in slot.get("allowed_thirds", [])])

        rows.append(
            {
                "match_id": slot["match_id"],
                "round": slot["round"],
                "slot_a": slot_a,
                "team_a": team_a,
                "slot_b_original": slot_b_original,
                "slot_b_resolved": slot_b_resolved,
                "team_b": team_b,
                "allowed_thirds": allowed,
                "date": slot.get("date", "TBD"),
                "venue_country": slot.get("venue_country", "TBD"),
            }
        )

    progress.value = 50
    status.value = "<b>[2/4]</b> Llave inicial construida."

    return pd.DataFrame(rows), third_assignment


def simulate_knockouts_with_progress(
    project_root: Path,
    group_stage: dict,
    third_assignment: dict,
    scenario: str,
    seed: int,
    progress,
    status,
):
    from src.simulation.tournament_simulator import R32_SLOTS, NEXT_ROUNDS, resolve_slot
    from src.simulation.knockout_match_simulator import simulate_knockout_match

    qualifiers = group_stage["qualifiers"]

    winners = {}
    losers = {}
    bracket_rows = []

    total_matches = len(R32_SLOTS) + len(NEXT_ROUNDS)
    current = 0

    rng = np.random.default_rng(seed + 100000)

    def run_one_match(slot, home, away):
        nonlocal current

        current += 1
        pct = int(52 + (current / max(total_matches, 1)) * 44)
        progress.value = min(pct, 96)
        status.value = (
            f"<b>[3/4]</b> Simulando {slot['round']} · "
            f"M{slot['match_id']}: {home} vs {away}"
        )

        result = simulate_knockout_match(
            home_team=home,
            away_team=away,
            match_date=slot.get("date", "2026-07-01"),
            round_name=slot["round"],
            venue_country=slot.get("venue_country", "TBD"),
            project_root=project_root,
            scenario=scenario,
            rng=rng,
            corners_cards_mode="legacy",
        )

        winners[slot["match_id"]] = result["winner"]
        losers[slot["match_id"]] = result["loser"]

        bracket_rows.append(
            {
                "match_id": slot["match_id"],
                "round": slot["round"],
                "home_team": home,
                "away_team": away,
                "home_goals_90": result.get("home_goals_90"),
                "away_goals_90": result.get("away_goals_90"),
                "home_goals_et": result.get("home_goals_et", 0),
                "away_goals_et": result.get("away_goals_et", 0),
                "winner": result["winner"],
                "loser": result["loser"],
                "method": result["method"],
                "scenario": scenario,
            }
        )

    for slot in R32_SLOTS:
        slot_a = slot["slot_a"]
        slot_b = slot["slot_b"]

        if slot_b == "3?":
            slot_b = third_assignment[slot["match_id"]]

        home = resolve_slot(slot_a, qualifiers)
        away = resolve_slot(slot_b, qualifiers)

        run_one_match(slot, home, away)

    for slot in NEXT_ROUNDS:
        if "from_a" in slot:
            home = winners[slot["from_a"]]
            away = winners[slot["from_b"]]
        else:
            home = losers[slot["loser_a"]]
            away = losers[slot["loser_b"]]

        run_one_match(slot, home, away)

    progress.value = 100
    status.value = "<b>[4/4]</b> Copa completada."

    bracket = pd.DataFrame(bracket_rows)

    return {
        "bracket": bracket,
        "champion": winners[104],
        "runner_up": losers[104],
        "third_place": winners[103],
    }


def lanzar_career_mode_menu(project_root: str | Path):
    if widgets is None:
        raise ImportError("ipywidgets no está disponible.")

    project_root = Path(project_root)

    state = {
        "group_stage": None,
        "r32": None,
        "third_assignment": None,
        "knockout": None,
    }

    fecha_corte = widgets.Text(
        value=str(pd.Timestamp.today().date()),
        description="Fecha corte:",
        layout=widgets.Layout(width="270px"),
    )

    escenario = widgets.Dropdown(
        options=["base", "moderate", "diaspora"],
        value="base",
        description="Escenario:",
        layout=widgets.Layout(width="240px"),
    )

    seed = widgets.IntText(
        value=2026,
        description="Seed:",
        layout=widgets.Layout(width="170px"),
    )

    usar_registro = widgets.Checkbox(
        value=True,
        description="Usar resultados registrados",
        indent=False,
        layout=widgets.Layout(width="260px"),
    )

    progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        bar_style="info",
        layout=widgets.Layout(width="70%"),
    )

    status = widgets.HTML(
        value="<b>Listo.</b> Elige una acción para comenzar.",
        layout=widgets.Layout(width="100%"),
    )

    out_copa = widgets.Output()
    out_grupos = widgets.Output()
    out_mc = widgets.Output()
    out_ko = widgets.Output()

    btn_sim_groups = widgets.Button(
        description="1. Simular grupos",
        button_style="success",
        layout=widgets.Layout(width="210px"),
    )

    btn_build_bracket = widgets.Button(
        description="2. Construir llave",
        button_style="info",
        layout=widgets.Layout(width="210px"),
    )

    btn_sim_knockouts = widgets.Button(
        description="3. Simular eliminatorias",
        button_style="warning",
        layout=widgets.Layout(width="230px"),
    )

    btn_full = widgets.Button(
        description="🏆 Simular Copa completa",
        button_style="danger",
        layout=widgets.Layout(width="250px"),
    )

    btn_reset = widgets.Button(
        description="Reiniciar copa",
        button_style="",
        layout=widgets.Layout(width="170px"),
    )

    def reset_state(_=None):
        state["group_stage"] = None
        state["r32"] = None
        state["third_assignment"] = None
        state["knockout"] = None
        progress.value = 0
        status.value = "<b>Reiniciado.</b> Puedes comenzar una nueva Copa."
        with out_copa:
            clear_output()
            display_html(warn_html("Copa reiniciada."))

    btn_reset.on_click(reset_state)

    def run_groups(_=None):
        with out_copa:
            clear_output()
            try:
                t0 = time.time()
                progress.value = 0
                status.value = "<b>[1/4]</b> Iniciando simulación de grupos..."

                group_stage = build_group_stage_with_progress(
                    project_root=project_root,
                    analysis_date=fecha_corte.value,
                    use_registered_results=bool(usar_registro.value),
                    seed=int(seed.value),
                    progress=progress,
                    status=status,
                )

                state["group_stage"] = group_stage
                state["r32"] = None
                state["third_assignment"] = None
                state["knockout"] = None

                elapsed = time.time() - t0
                progress.value = 45
                status.value = f"<b>[1/4]</b> Grupos listos en {elapsed:.1f} s."

                display_html(ok_html(f"Fase de grupos simulada en {elapsed:.1f} s."))
                display_html(compact_group_tables_html(group_stage))

                display_html("<h3>Mejores terceros</h3>")
                cols = ["third_rank", "Grupo", "Equipo", "Pts", "DG", "GF", "qualifies_as_best_third"]
                cols = [c for c in cols if c in group_stage["best_thirds"].columns]
                display(group_stage["best_thirds"][cols])

            except Exception as e:
                display_html(error_html(e))
                status.value = "<b>Error.</b> La simulación de grupos falló."

    btn_sim_groups.on_click(run_groups)

    def run_build_bracket(_=None):
        with out_copa:
            try:
                if state["group_stage"] is None:
                    display_html(warn_html("Primero simula la fase de grupos."))
                    return

                r32, third_assignment = build_r32_bracket_with_progress(
                    group_stage=state["group_stage"],
                    progress=progress,
                    status=status,
                )

                state["r32"] = r32
                state["third_assignment"] = third_assignment

                display_html(ok_html("Llave de ronda de 32 construida automáticamente."))
                display_html(render_r32_cards(r32))

            except Exception as e:
                display_html(error_html(e))
                status.value = "<b>Error.</b> No se pudo construir la llave."

    btn_build_bracket.on_click(run_build_bracket)

    def run_knockouts(_=None):
        with out_copa:
            try:
                if state["group_stage"] is None:
                    display_html(warn_html("Primero simula la fase de grupos."))
                    return

                if state["third_assignment"] is None:
                    r32, third_assignment = build_r32_bracket_with_progress(
                        group_stage=state["group_stage"],
                        progress=progress,
                        status=status,
                    )
                    state["r32"] = r32
                    state["third_assignment"] = third_assignment
                    display_html(render_r32_cards(r32))

                t0 = time.time()

                knockout = simulate_knockouts_with_progress(
                    project_root=project_root,
                    group_stage=state["group_stage"],
                    third_assignment=state["third_assignment"],
                    scenario=escenario.value,
                    seed=int(seed.value),
                    progress=progress,
                    status=status,
                )

                state["knockout"] = knockout

                elapsed = time.time() - t0

                display_html(render_final_summary(
                    champion=knockout["champion"],
                    runner_up=knockout["runner_up"],
                    third_place=knockout["third_place"],
                ))

                display_html(ok_html(f"Eliminatorias simuladas en {elapsed:.1f} s."))
                display_bracket_table(knockout["bracket"])

            except Exception as e:
                display_html(error_html(e))
                status.value = "<b>Error.</b> La simulación de eliminatorias falló."

    btn_sim_knockouts.on_click(run_knockouts)

    def run_full_cup(_=None):
        with out_copa:
            clear_output()
            try:
                reset_state()
                t0 = time.time()

                group_stage = build_group_stage_with_progress(
                    project_root=project_root,
                    analysis_date=fecha_corte.value,
                    use_registered_results=bool(usar_registro.value),
                    seed=int(seed.value),
                    progress=progress,
                    status=status,
                )
                state["group_stage"] = group_stage

                display_html(ok_html("Fase de grupos terminada."))
                display_html(compact_group_tables_html(group_stage))

                r32, third_assignment = build_r32_bracket_with_progress(
                    group_stage=group_stage,
                    progress=progress,
                    status=status,
                )
                state["r32"] = r32
                state["third_assignment"] = third_assignment

                display_html(render_r32_cards(r32))

                knockout = simulate_knockouts_with_progress(
                    project_root=project_root,
                    group_stage=group_stage,
                    third_assignment=third_assignment,
                    scenario=escenario.value,
                    seed=int(seed.value),
                    progress=progress,
                    status=status,
                )
                state["knockout"] = knockout

                elapsed = time.time() - t0

                display_html(render_final_summary(
                    champion=knockout["champion"],
                    runner_up=knockout["runner_up"],
                    third_place=knockout["third_place"],
                ))

                display_html(ok_html(f"Copa completa simulada en {elapsed:.1f} s."))
                display_bracket_table(knockout["bracket"])

            except Exception as e:
                display_html(error_html(e))
                status.value = "<b>Error.</b> La Copa completa falló."

    btn_full.on_click(run_full_cup)

    # ------------------------------------------------------------
    # Pestaña Grupos: no se mueve el selector anterior.
    # ------------------------------------------------------------
    btn_open_group_selector = widgets.Button(
        description="Abrir selector de grupos",
        button_style="success",
        layout=widgets.Layout(width="260px"),
    )

    def open_group_selector(_):
        with out_grupos:
            clear_output()
            try:
                from src.interactive.group_match_selector_v2 import lanzar_selector_grupos_v2
                display_html(
                    "<div style='padding:12px;border-radius:12px;background:#F8FAFC;"
                    "border:1px solid #EAECF0;'>Se abre el selector de grupos que ya tenías.</div>"
                )
                lanzar_selector_grupos_v2(project_root=project_root, corners_cards_mode="legacy")
            except Exception as e:
                display_html(error_html(e))

    btn_open_group_selector.on_click(open_group_selector)

    tab_grupos = widgets.VBox(
        [
            note_widget("Esta pestaña mantiene el selector de grupos anterior. No se modifica su lógica."),
            btn_open_group_selector,
            out_grupos,
        ]
    )

    # ------------------------------------------------------------
    # Pestaña Modo Copa.
    # ------------------------------------------------------------
    tab_copa = widgets.VBox(
        [
            note_widget(
                "Flujo recomendado: simular grupos → construir llave → simular eliminatorias. "
                "La llave resuelve automáticamente los mejores terceros."
            ),
            widgets.HBox([fecha_corte, escenario, seed, usar_registro]),
            widgets.HBox([progress]),
            status,
            widgets.HBox([btn_sim_groups, btn_build_bracket, btn_sim_knockouts, btn_full, btn_reset]),
            out_copa,
        ]
    )

    # ------------------------------------------------------------
    # Pestaña Campeón Monte Carlo.
    # ------------------------------------------------------------
    mc_preset = widgets.Dropdown(
        options=[
            ("Prueba mínima — 3 Copas", 3),
            ("Prueba — 5 Copas", 5),
            ("Rápido — 10 Copas", 10),
            ("Medio — 50 Copas", 50),
            ("Más serio — 100 Copas", 100),
        ],
        value=5,
        description="Preset:",
        layout=widgets.Layout(width="300px"),
    )

    mc_n = widgets.IntText(
        value=5,
        description="N:",
        layout=widgets.Layout(width="150px"),
    )

    mc_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        bar_style="info",
        layout=widgets.Layout(width="70%"),
    )

    mc_status = widgets.HTML(value="<b>Listo.</b> Elige N y corre Monte Carlo.")

    btn_mc = widgets.Button(
        description="Simular campeones",
        button_style="success",
        layout=widgets.Layout(width="220px"),
    )

    def sync_mc(change):
        mc_n.value = int(change["new"])

    mc_preset.observe(sync_mc, names="value")

    def run_mc(_):
        with out_mc:
            clear_output()
            try:
                from src.simulation.tournament_simulator import simulate_tournament_once

                n = int(mc_n.value)
                rows = []
                t0 = time.time()

                for i in range(n):
                    pct = int((i / max(n, 1)) * 100)
                    mc_progress.value = pct
                    mc_status.value = f"<b>Monte Carlo:</b> simulando Copa {i + 1} de {n}..."

                    out = simulate_tournament_once(
                        project_root=project_root,
                        analysis_date=fecha_corte.value,
                        use_registered_results=bool(usar_registro.value),
                        seed=int(seed.value) + i,
                        scenario=escenario.value,
                        corners_cards_mode="legacy",
                    )

                    rows.append(
                        {
                            "simulation": i + 1,
                            "champion": out["champion"],
                            "runner_up": out["runner_up"],
                            "third_place": out["third_place"],
                            "scenario": escenario.value,
                        }
                    )

                mc_progress.value = 100
                elapsed = time.time() - t0
                mc_status.value = f"<b>Monte Carlo terminado.</b> Tiempo: {elapsed:.1f} s."

                sims = pd.DataFrame(rows)

                probs = (
                    sims["champion"]
                    .value_counts(normalize=True)
                    .reset_index()
                )
                probs.columns = ["team", "champion_probability"]
                counts = sims["champion"].value_counts()
                probs["champion_count"] = probs["team"].map(counts)
                probs = probs.sort_values("champion_probability", ascending=False).reset_index(drop=True)

                display_html(ok_html(f"Monte Carlo terminado: {n} Copas en {elapsed:.1f} s."))
                display(probs.head(25))

            except Exception as e:
                display_html(error_html(e))
                mc_status.value = "<b>Error.</b> Falló el Monte Carlo."

    btn_mc.on_click(run_mc)

    tab_mc = widgets.VBox(
        [
            note_widget(
                "Monte Carlo de Copas completas. Empieza con 3 o 5 Copas para validar. "
                "Después subimos N cuando optimicemos velocidad."
            ),
            widgets.HBox([fecha_corte, escenario, seed, usar_registro]),
            widgets.HBox([mc_preset, mc_n]),
            widgets.HBox([mc_progress]),
            mc_status,
            btn_mc,
            out_mc,
        ]
    )

    # ------------------------------------------------------------
    # Pestaña KO libre.
    # ------------------------------------------------------------
    ko_home = widgets.Text(value="Mexico", description="Equipo 1:", layout=widgets.Layout(width="300px"))
    ko_away = widgets.Text(value="France", description="Equipo 2:", layout=widgets.Layout(width="300px"))
    ko_round = widgets.Dropdown(
        options=["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"],
        value="Quarterfinal",
        description="Ronda:",
        layout=widgets.Layout(width="250px"),
    )
    ko_date = widgets.Text(value="2026-07-09", description="Fecha:", layout=widgets.Layout(width="230px"))
    ko_venue = widgets.Text(value="United States", description="Sede:", layout=widgets.Layout(width="260px"))

    btn_ko = widgets.Button(
        description="Simular KO libre",
        button_style="warning",
        layout=widgets.Layout(width="220px"),
    )

    def run_ko_free(_):
        with out_ko:
            clear_output()
            try:
                from src.simulation.knockout_match_simulator import simulate_knockout_match

                display_html(warn_html(
                    "Este modo es libre/hipotético. Para la llave real simulada, usa la pestaña Modo Copa."
                ))

                t0 = time.time()

                result = simulate_knockout_match(
                    home_team=ko_home.value,
                    away_team=ko_away.value,
                    match_date=ko_date.value,
                    round_name=ko_round.value,
                    venue_country=ko_venue.value,
                    project_root=project_root,
                    scenario=escenario.value,
                    seed=int(seed.value),
                    corners_cards_mode="legacy",
                )

                elapsed = time.time() - t0

                display_html(ok_html(f"Ganador: {team_label(result['winner'])} · Tiempo: {elapsed:.1f} s."))

                cols = [
                    "home_team",
                    "away_team",
                    "round_name",
                    "winner",
                    "method",
                    "home_goals_90",
                    "away_goals_90",
                    "home_goals_et",
                    "away_goals_et",
                    "scenario",
                    "net_host_advantage",
                ]
                cols = [c for c in cols if c in result]
                display(pd.DataFrame([{c: result.get(c) for c in cols}]))

            except Exception as e:
                display_html(error_html(e))

    btn_ko.on_click(run_ko_free)

    tab_ko = widgets.VBox(
        [
            note_widget(
                "Modo experimental para probar una eliminatoria hipotética. "
                "No resuelve slots de mejores terceros; eso lo hace Modo Copa."
            ),
            widgets.HBox([ko_home, ko_away]),
            widgets.HBox([ko_round, ko_date, ko_venue]),
            widgets.HBox([escenario, seed]),
            btn_ko,
            out_ko,
        ]
    )

    tabs = widgets.Tab(children=[tab_grupos, tab_copa, tab_mc, tab_ko])
    titles = ["Grupos", "Modo Copa", "Campeón MC", "KO libre"]

    for i, title in enumerate(titles):
        tabs.set_title(i, title)

    display(header_widget())
    display(tabs)
