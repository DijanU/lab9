# Laboratorio 9 — Series de tiempo y forecasting

CC3092 Deep Learning y Sistemas Inteligentes. Pronóstico *day-ahead* (H = 24 h) del consumo eléctrico horario de 100 clientes de Portugal (ElectricityLoadDiagrams20112014, UCI).

| Archivo | Contenido |
|---|---|
| `investigacion.md` | Investigación (sección 3): `nn.LSTM`/`nn.GRU`, `Conv1d` dilatada y causal, `TransformerEncoderLayer`, NeuralForecast/StatsForecast, Chronos, y los conceptos (local contra global, teacher forcing, RevIN, pinball loss, MASE contra WAPE). |
| `lab9_forecasting.ipynb` | Notebook completo y ejecutado: limpieza, exploración, `Dataset` de ventanas, auditoría de fugas, `backtest`, modelos propios, librerías, iteraciones y evaluación final. |
| `lab9_src.py` | Fuente del notebook (celdas `# %%`); `build_notebook.py` lo convierte en `.ipynb`. |
| `informe.md` / `informe.pdf` | Informe final. |
| `results/pronosticos_prueba.parquet` | Pronósticos de prueba de cada modelo: `unique_id, ds, cutoff, modelo, y_hat` (H = 24, 92 orígenes). `results/reales_prueba.parquet` tiene los valores reales. |
| `results/`, `figures/` | Tablas (CSV/JSON) y figuras generadas por el notebook. |

## Reproducir

```bash
python3.14 -m venv .venv
.venv/bin/pip install torch numpy pandas matplotlib scikit-learn lightgbm statsforecast neuralforecast chronos-forecasting holidays pyarrow jupyter nbformat
# macOS: LightGBM y PyTorch traen copias distintas de libomp y el proceso termina en segfault al
# usar ambos. Se enlaza LightGBM al libomp de torch:
SP=.venv/lib/python3.14/site-packages
install_name_tool -change @rpath/libomp.dylib "$PWD/$SP/torch/lib/libomp.dylib" $SP/lightgbm/lib/lib_lightgbm.dylib
codesign --force -s - $SP/lightgbm/lib/lib_lightgbm.dylib
.venv/bin/python build_notebook.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 lab9_forecasting.ipynb
```

El notebook descarga los datos en `data/`. Cada experimento guarda su resultado en `cache/` y, al volver a ejecutar, lo reutiliza. `LAB9_FAST=1` corre una versión reducida (20 clientes, pocas epochs) para probar el pipeline.

Hardware: MacBook Pro con Apple M5 Pro. Las redes corren en PyTorch MPS; StatsForecast y LightGBM corren en CPU, en un hilo paralelo al entrenamiento en GPU.
