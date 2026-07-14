# Phase 5.1.2 — Bleed-through + fondos reales (v2, con hoja en blanco)

REEMPLAZA al zip phase5_1_2 anterior si lo descargaste: mismo código base
más el fix de banda del cosechador, el flag --dark-offset y 163 tiles
(antes 31).

Contenido:
- data/generators/degradations.py     (add_bleedthrough)
- data/generate_dataset.py            (rama real-bg + bleed-through + CLI)
- scripts/harvest_backgrounds.py      (NUEVO, con --dark-offset)
- tests/ (test_generators, test_dataset, test_harvest_backgrounds NUEVO)
- data/real_backgrounds/              (163 tiles + .gitkeep):
    31  bg_Escáner/WhatsApp_*  — páginas de cómic dibujadas, escala nativa
    82  bg_blank600_*          — hoja en blanco, 600 DPI, tiles 512
    50  bg_blank180_*          — hoja en blanco reescalada a DPI de cómic
- contact_sheet.jpg                   (verificación visual de los 163)
- _gitignore_updated                  (renombrar a .gitignore)

## Aplicación (PowerShell, desde C:\Users\oliju\Documents\DocClean-Net)

IMPORTANTE: aplica ANTES el zip phase5_1_1_archetype si no lo hiciste
(5.1.2 se construye encima; el commit f662063 solo contiene 5.1).

```powershell
$src = "$env:USERPROFILE\Downloads\phase5_1_2"
Copy-Item "$src\data\generators\degradations.py" data\generators\ -Force
Copy-Item "$src\data\generate_dataset.py" data\ -Force
Copy-Item "$src\scripts\harvest_backgrounds.py" scripts\
Copy-Item "$src\tests\*.py" tests\ -Force
New-Item -ItemType Directory -Force data\real_backgrounds | Out-Null
Copy-Item "$src\data\real_backgrounds\*" data\real_backgrounds\
Copy-Item "$src\_gitignore_updated" .gitignore -Force
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
```

Esperado: 233 passed, 4 deselected.

```powershell
git add data tests scripts .gitignore
git commit -m "feat: Phase 5.1.2 - bleed-through degradation + real background compositing (harvester, 163 tiles incl. blank-page scan at two scales)"
git push
```

Los tiles quedan gitignorados (solo viaja el .gitkeep). Consérvalos en
data\real_backgrounds\ — son parte del pipeline de generación local.

## Generación del dataset v1.1 (para la sesión de Colab/Kaggle)

```
python -m data.generate_dataset --n 10000 --output data/synthetic/ \
    --size 512 --seed 51 --workers 4 --domain-robust-prob 0.5 \
    --real-bg-dir data/real_backgrounds --real-bg-prob 0.15
```

Los tiles no están en el repo: comprímelos (~12 MB) y súbelos a la sesión.
Cuando confirmes el push te entrego el notebook actualizado con la celda de
subida/descompresión y este comando.

## Si escaneas más páginas en blanco

```powershell
.\.venv\Scripts\python.exe -m scripts.harvest_backgrounds `
    --input ruta\a\escaneos_en_blanco `
    --output data\real_backgrounds `
    --tile 512 --stride 256 --dark-offset 40
```

Verifica siempre contact_sheet.jpg y borra a mano lo dudoso. En páginas
DIBUJADAS usa el default --dark-offset 60 (más estricto).
