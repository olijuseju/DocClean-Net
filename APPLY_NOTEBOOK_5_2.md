# Phase 5.2 — Notebook de Colab actualizado: aplicación (PowerShell)

Desde C:\Users\oliju\Documents\DocClean-Net, con el .ipynb descargado en Downloads:

```powershell
Copy-Item "$env:USERPROFILE\Downloads\02_training_colab.ipynb" notebooks\ -Force
git add notebooks\02_training_colab.ipynb ; git commit -m "fix: Colab notebook - restore missing clone cell, remove phantom --offset flag and duplicate dataset cell; feat: v1.1 domain-robust dataset generation" ; git push
```

## Bugs corregidos en el notebook (además de la actualización v1.1)

1. **Celda de clonado ausente**: bajo el header "1 · Clonar repositorio" había
   una celda de generación de dataset. En un Colab limpio el notebook moría en
   la celda de dependencias con ModuleNotFoundError. Restaurada (clone
   idempotente + %cd).
2. **Flag fantasma `--offset`**: la celda incremental usaba un argumento que
   generate_dataset.py nunca ha tenido — argparse habría abortado. Eliminada.
3. **Celda de dataset duplicada** con lógica contradictoria. Queda una sola.
4. **`%cd /content/DocClean-Net` en todas las celdas de código** (lección de
   Phase 2: los paths relativos fallan en silencio sin guard).

## Cambios v1.1

- Dataset: `--seed 51 --domain-robust-prob 0.5` (10k pares, 512px). Seed nueva
  para trazabilidad del dataset de Phase 5.
- Entrenamiento: receta idéntica a v1.0 (50 épocas, batch 16, lr 1e-3,
  patch 256). Si val_loss sigue bajando en la época 50, se relanza más largo —
  decisión con train_log.csv en la mano.

## Flujo de la sesión de Colab

1. Ejecutar todo el notebook (GPU T4). La generación de 10k pares tarda unos
   minutos con 4 workers; el entrenamiento, similar a v1.0 (~misma época/min).
2. Descargar `best.pt` y `train_log.csv` (celda 7).
3. Renombrar el log localmente antes de commitear para no pisar el de v1.0:
   `checkpoints/train_log_v1_1.csv` (el best.pt sigue gitignorado).
4. Enviarme `train_log_v1_1.csv` (o pegar las últimas ~10 filas) y los
   resultados de la celda 8 → decidimos si el checkpoint pasa a evaluación
   (benchmark + los 2 casos del README + el set de 18) o si relanzamos.
