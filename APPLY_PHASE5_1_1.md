# Phase 5.1.1 — Arquetipo milimetrado: aplicación (PowerShell)

Motivación (medida, no opinión): con el muestreo independiente de 5.1, el
combo de fallo real de los cómics (grid denso + oscuro + tinta tenue) caía
en ~0.6% del dataset (≈56 imágenes de 10k) — señal insuficiente. Los ejes
correlacionan en el papel real: son arquetipos, no ruletas independientes.

Cambios:
- `generate_dataset.py`: arquetipo "milimetrado impreso" con prob. 0.35
  dentro del régimen robust — spacing [10,20), gris [60,100), opacity
  [0.85,1.0], blend opaco, tinta tenue al 70%, siempre cuadrícula. El
  muestreo independiente se mantiene como fondo de diversidad. Además, la
  pauta (ruled) gana tratamiento oscuro con prob. 0.35 en robust.
- `degradations.py`: `add_ruled_lines` gana `opaque_lines` (mismo blend de
  solo-oscurecimiento que la grid).
- Tests: +5 (arquetipo, pauta opaca, cobertura del combo en lote).

Cobertura medida tras el cambio (500 pares, robust_prob=0.5):
combo cómic 0.6% → **10.4%** (~1040 imágenes de 10k); banda oscura 13.2%.

Desde C:\Users\oliju\Documents\DocClean-Net, con el zip en Downloads\phase5_1_1:

```powershell
$src = "$env:USERPROFILE\Downloads\phase5_1_1"
Copy-Item "$src\data\generators\degradations.py" data\generators\ -Force ; Copy-Item "$src\data\generate_dataset.py" data\ -Force ; Copy-Item "$src\tests\test_generators.py" tests\ -Force ; Copy-Item "$src\tests\test_dataset.py" tests\ -Force
```

Verificación (esperado: 212 passed, 4 deselected):

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not slow" -q
```

Commit:

```powershell
git add data tests ; git commit -m "feat: Phase 5.1.1 - correlated 'printed graph paper' archetype sampling (comic failure combo 0.6% -> 10.4% of dataset), dark ruled lines" ; git push
```

El notebook de Colab NO cambia: mismo comando de generación
(--seed 51 --domain-robust-prob 0.5); el arquetipo vive dentro del generador.
