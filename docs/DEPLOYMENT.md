# Despliegue online

La app está preparada para publicarse con GitHub + Streamlit Community Cloud.

## Prueba local

python -m py_compile app.py
streamlit run app.py

## Publicación

1. Subir el repo a GitHub.
2. Entrar a Streamlit Community Cloud.
3. Seleccionar el repo.
4. Main file path: app.py.
5. Deploy.

## No subir

- .venv/
- data/raw/
- data/scenarios/
- data/predictions/formatted/
- data/predictions/mc_original_v2_runs/
