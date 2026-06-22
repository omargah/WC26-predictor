# -*- coding: utf-8 -*-

"""
scripts/build_annexe_c_v2.py

Extrae Annexe C del reglamento oficial FIFA World Cup 2026.

Salida:
    data/external/fifa_2026_annexe_c_third_place_combinations.csv

La tabla debe tener:
    - 495 opciones
    - 8 columnas de asignación:
        1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L
    - una llave third_groups_key, por ejemplo: BCDEFHKL
"""

from pathlib import Path
import re
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from pypdf import PdfReader

from src.config import get_paths


FIFA_REGULATIONS_URL = (
    "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/"
    "FWC2026_regulations_EN.pdf"
)

ANNEXE_COLUMNS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


def download_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0:
        print(f"PDF ya existe: {path}")
        return

    print("Descargando reglamento FIFA desde:")
    print(FIFA_REGULATIONS_URL)

    urllib.request.urlretrieve(FIFA_REGULATIONS_URL, path)


def extract_text_with_pypdf(path: Path) -> str:
    reader = PdfReader(str(path))

    parts = []

    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        parts.append(f"\n\n===== PAGE {i} =====\n\n")
        parts.append(txt)

    return "\n".join(parts)


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("\u2007", " ")
    text = text.replace("\u202f", " ")
    text = text.replace("\t", " ")
    return text


def parse_annexe_c_global(text: str) -> pd.DataFrame:
    """
    Parser global.

    Busca patrones como:
        1 3E 3J 3I 3F 3H 3G 3L 3K

    También tolera:
        1 3 E 3 J 3 I ...
    """

    text = normalize_text(text)

    code = r"3\s*([A-L])"

    pattern = re.compile(
        rf"(?<!\d)"
        rf"(\d{{1,3}})\s+"
        rf"{code}\s+"
        rf"{code}\s+"
        rf"{code}\s+"
        rf"{code}\s+"
        rf"{code}\s+"
        rf"{code}\s+"
        rf"{code}\s+"
        rf"{code}"
        rf"(?!\d)",
        flags=re.MULTILINE,
    )

    rows_by_option = {}

    for match in pattern.finditer(text):
        option = int(match.group(1))

        if option < 1 or option > 495:
            continue

        groups = list(match.groups()[1:])
        values = [f"3{g}" for g in groups]

        if len(set(values)) != 8:
            continue

        if option in rows_by_option:
            if rows_by_option[option] != values:
                # Si aparece duplicado por encabezados o reflujo extraño,
                # conservamos el primero y lo reportamos solo si al final falla.
                pass
            continue

        rows_by_option[option] = values

    return rows_from_options(rows_by_option)


def parse_annexe_c_by_lines(text: str) -> pd.DataFrame:
    """
    Parser por líneas, útil cuando el texto extraído conserva filas.
    """

    text = normalize_text(text)

    rows_by_option = {}

    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        tokens = line.split()

        # Caso normal:
        # 1 3E 3J 3I 3F 3H 3G 3L 3K
        if len(tokens) >= 9 and tokens[0].isdigit():
            option = int(tokens[0])

            if 1 <= option <= 495:
                values = []

                for tok in tokens[1:]:
                    tok = tok.strip()

                    if re.fullmatch(r"3[A-L]", tok):
                        values.append(tok)

                    if len(values) == 8:
                        break

                if len(values) == 8 and len(set(values)) == 8:
                    rows_by_option.setdefault(option, values)

        # Caso separado:
        # 1 3 E 3 J 3 I ...
        m = re.match(r"^(\d{1,3})\s+(.*)$", line)

        if not m:
            continue

        option = int(m.group(1))

        if not (1 <= option <= 495):
            continue

        groups = re.findall(r"3\s*([A-L])", m.group(2))

        if len(groups) >= 8:
            values = [f"3{g}" for g in groups[:8]]

            if len(set(values)) == 8:
                rows_by_option.setdefault(option, values)

    return rows_from_options(rows_by_option)


def rows_from_options(rows_by_option: dict[int, list[str]]) -> pd.DataFrame:
    rows = []

    for option in sorted(rows_by_option):
        values = rows_by_option[option]

        third_groups = sorted({v.replace("3", "") for v in values})

        if len(third_groups) != 8:
            continue

        row = {"option": option}

        for col, value in zip(ANNEXE_COLUMNS, values):
            row[col] = value

        row["third_groups_key"] = "".join(third_groups)

        rows.append(row)

    return pd.DataFrame(rows)


def validate_annexe_c(df: pd.DataFrame) -> None:
    if len(df) != 495:
        raise RuntimeError(
            f"Annexe C debería tener 495 opciones y se extrajeron {len(df)}."
        )

    if df["option"].nunique() != 495:
        duplicated = df[df["option"].duplicated(keep=False)]
        raise RuntimeError(
            "Hay opciones duplicadas. Ejemplos:\n"
            + duplicated.head(20).to_string(index=False)
        )

    if df["third_groups_key"].nunique() != 495:
        duplicated = df[df["third_groups_key"].duplicated(keep=False)]
        raise RuntimeError(
            "Hay third_groups_key duplicados. Ejemplos:\n"
            + duplicated.head(20).to_string(index=False)
        )

    expected_cols = ["option"] + ANNEXE_COLUMNS + ["third_groups_key"]
    missing = [c for c in expected_cols if c not in df.columns]

    if missing:
        raise RuntimeError("Faltan columnas: " + ", ".join(missing))

    for col in ANNEXE_COLUMNS:
        bad = ~df[col].astype(str).str.fullmatch(r"3[A-L]")

        if bad.any():
            raise RuntimeError(
                f"Valores inválidos en columna {col}:\n"
                + df.loc[bad, ["option", col]].head(20).to_string(index=False)
            )


def main() -> None:
    paths = get_paths()

    external_dir = paths["data"] / "external"
    external_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = external_dir / "FWC2026_regulations_EN.pdf"
    text_path = external_dir / "FWC2026_regulations_text_dump.txt"
    out_path = external_dir / "fifa_2026_annexe_c_third_place_combinations.csv"

    download_pdf(pdf_path)

    text = extract_text_with_pypdf(pdf_path)
    text = normalize_text(text)

    text_path.write_text(text, encoding="utf-8")

    parsers = [
        ("global", parse_annexe_c_global),
        ("line_by_line", parse_annexe_c_by_lines),
    ]

    last_error = None

    for parser_name, parser in parsers:
        print()
        print(f"Intentando parser: {parser_name}")

        df = parser(text)

        print(f"Filas extraídas: {len(df)}")

        try:
            validate_annexe_c(df)
        except Exception as e:
            last_error = e
            print(f"Parser {parser_name} falló: {e}")
            continue

        df = df.sort_values("option").reset_index(drop=True)
        df.to_csv(out_path, index=False, encoding="utf-8")

        print()
        print("=" * 90)
        print("ANNEXE C FIFA 2026 EXTRAÍDO CORRECTAMENTE")
        print("=" * 90)
        print(f"parser:   {parser_name}")
        print(f"PDF:      {pdf_path}")
        print(f"text:     {text_path}")
        print(f"CSV:      {out_path}")
        print()
        print(df.head(10).to_string(index=False))
        print()
        print(df.tail(10).to_string(index=False))
        print()
        print("=" * 90)

        return

    print()
    print("=" * 90)
    print("NO SE PUDO EXTRAER ANNEXE C")
    print("=" * 90)
    print(f"Texto extraído guardado en: {text_path}")
    print("Abre ese archivo y busca: Annexe C")
    print("Último error:")
    print(last_error)
    print("=" * 90)

    raise RuntimeError(
        "No se pudo extraer Annexe C automáticamente. "
        f"Revisa el texto en {text_path}"
    )


if __name__ == "__main__":
    main()
