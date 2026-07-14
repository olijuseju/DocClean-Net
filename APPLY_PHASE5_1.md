# Phase 5.1 — Generador domain-robust: instrucciones de aplicación (PowerShell)

Desde la raíz del repo (C:\Documents\Proyectos\DocClean-Net), con el zip
descomprimido en $env:USERPROFILE\Downloads\phase5_1:

```powershell
$src = "$env:USERPROFILE\Downloads\phase5_1"
Copy-Item "$src\data\generators\illumination.py" data\generators\ ; Copy-Item "$src\data\generators\strokes.py" data\generators\ -Force ; Copy-Item "$src\data\generators\degradations.py" data\generators\ -Force ; Copy-Item "$src\data\generate_dataset.py" data\ -Force ; Copy-Item "$src\tests\test_illumination.py" tests\ ; Copy-Item "$src\tests\test_generators.py" tests\ -Force ; Copy-Item "$src\tests\test_dataset.py" tests\ -Force
```

Housekeeping aprobado en el plan (artefacto commiteado por error en af6f3d2):

```powershell
git rm phase4-tasks1-2-changes.patch
```

Verificación (esperado: 207 passed, 4 deselected; black y ruff limpios):

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q ; .\.venv\Scripts\python.exe -m black --check data tests ; .\.venv\Scripts\python.exe -m ruff check data tests
```

Commit sugerido:

```powershell
git add data tests ; git commit -m "feat: Phase 5.1 - domain-robust synthetic generator (illumination fields, dark/dense grids, faint ink, stains)"
```

## Qué cambia

- **NUEVO `data/generators/illumination.py`** — campos de ganancia multiplicativos
  (linear/corner/radial/smooth), min_gain muestreado en [0.40, 0.85]. Se aplica
  SOLO al dirty; el target queda limpio → la red aprende a normalizar el fondo.
- **`data/generators/strokes.py`** — `generate_strokes(..., ink_color=None)`:
  None = negro histórico; grises simulan lápiz tenue (real medido: 64-92).
- **`data/generators/degradations.py`** — `add_blue_grid` gana `color_bgr` y
  `opaque_lines` (blend de solo-oscurecimiento con peso completo, necesario
  porque el blend histórico satura en peso 0.35 y el milimetrado real está en
  gris ≈85); nueva `add_stain` (manchas multiplicativas, tinta intacta).
- **`data/generate_dataset.py`** — `generate_pair`/`generate_dataset`/CLI ganan
  `domain_robust_prob` (default 0.5): mezcla 50/50 v1.0 / domain-robust.
  Rangos robust calibrados con el set real: spacing [10,46), grid gris [60,125)
  con sesgo azul ≤16, opacity [0.55,1.0], tinta tenue 40% en [60,160),
  iluminación 70%, manchas 25%.
- **Tests**: +37 (14 illumination, 17 generators, 6 dataset) → 207 fast en verde.
- **git rm phase4-tasks1-2-changes.patch** — artefacto huérfano.

## Nota de reproducibilidad

`generate_pair` consume una tirada extra del rng (la moneda robust), así que un
dataset v1.1 con la misma seed NO es byte-idéntico a uno v1.0 — está documentado
y testeado (`test_generate_pair_robust_prob_changes_output_for_same_seed`).
Sigue siendo determinista: misma seed + mismo domain_robust_prob → mismo dataset.

## Siguiente paso (5.2)

Regenerar dataset (10k pares, `--domain-robust-prob 0.5`), reentrenar en Colab
con la misma receta, y evaluar contra las puertas de aceptación del plan.
