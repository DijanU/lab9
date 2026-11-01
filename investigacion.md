# Investigación: herramientas para forecasting en PyTorch

Laboratorio 9 — CC3092 Deep Learning y Sistemas Inteligentes. Sección 3 del enunciado.

## Clases de PyTorch

### `torch.nn.LSTM` y `torch.nn.GRU`

Redes recurrentes que recorren la secuencia paso a paso y mantienen un estado oculto. La LSTM tiene además una celda de memoria `c` con tres compuertas (entrada, olvido y salida), que le permite conservar información durante muchos pasos. La GRU combina celda y estado en uno solo, con dos compuertas (actualización y reinicio): tiene ≈ 25 % menos parámetros y suele rendir parecido.

| Parámetro | Propósito |
|---|---|
| `input_size` | Número de variables por paso de tiempo (en este laboratorio, 8: consumo y 7 covariables de calendario). |
| `hidden_size` | Dimensión del estado oculto `h` (y de la celda `c`). Controla la capacidad. |
| `num_layers` | Capas recurrentes apiladas: la salida de una es la entrada de la siguiente. |
| `batch_first` | Si es `True`, la entrada y `output` tienen forma `(batch, seq, feature)`. Por defecto es `(seq, batch, feature)`. **No** afecta a `h_n` ni a `c_n`. |
| `dropout` | Dropout sobre la salida de cada capa **excepto la última**. Solo tiene efecto si `num_layers > 1`. |
| `bidirectional` | Recorre también en sentido inverso (`D = 2`). No sirve para pronóstico causal sobre el futuro, pero sí para codificar la ventana pasada. |
| `proj_size` (solo LSTM) | Proyecta `h` a una dimensión menor. |

**Formas de salida** (con `batch_first=True`, `D = 2` si es bidireccional, si no `D = 1`):

- `output`: `(B, L, D·hidden_size)`, el estado oculto de **la última capa** en **cada** paso. Se usa para seq2seq, para la estrategia autorregresiva entrenada con *teacher forcing* (una predicción por paso) o para atención.
- `h_n`: `(D·num_layers, B, hidden_size)`, el estado oculto **final** de **cada capa**. `h_n[-1]` es el resumen de la ventana que usa la cabeza multi-salida.
- `c_n` (solo LSTM): `(D·num_layers, B, hidden_size)`, la celda final de cada capa. La GRU devuelve solo `(output, h_n)`.

`(h_n, c_n)` puede pasarse como estado inicial de otra llamada. Así funcionan el decoder de un seq2seq y la generación autorregresiva paso a paso.

### `torch.nn.Conv1d` con `dilation` y padding causal (TCN)

`nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")` recibe `(B, C_in, L)` y desliza un filtro de tamaño `k` sobre el tiempo.

- **`dilation = d`** separa los coeficientes del filtro `d` pasos: el filtro mira `t, t−d, …, t−(k−1)d`. Así el campo receptivo crece sin agregar parámetros.
- **Padding causal.** Para que la salida en `t` solo dependa de `≤ t` se rellena **solo a la izquierda** con `(k−1)·d` ceros. Puede hacerse con `F.pad(x, ((k−1)·d, 0))` antes de una convolución con `padding=0`, o con `padding=(k−1)·d` y recortando los últimos `(k−1)·d` pasos de la salida. El padding simétrico de `padding="same"` filtra información del futuro.
- **Campo receptivo de una TCN.** Con `n` capas de kernel `k` y dilataciones `d_i`, el campo receptivo es

  $$R = 1 + \sum_i (k-1)\,d_i .$$

  - Con dilataciones dobladas en cada capa (`d_i = 2^i`): $R = 1 + (k-1)(2^n - 1)$.
  - En una TCN con bloques residuales de dos convoluciones por bloque: $R = 1 + 2(k-1)\sum_b d_b$.
  - Ejemplos con `k = 2` y `d = 1, 2, …, 32` (6 bloques): $R = 64$ con una convolución por bloque y $R = 127$ con dos.
  - Para cubrir `L = 168` hace falta, por ejemplo, `k = 3` con dos convoluciones por bloque: $R = 1 + 4\cdot 63 = 253$.

  Es decir, crece exponencialmente con la profundidad y linealmente con `k`.

### `torch.nn.TransformerEncoderLayer`

Bloque de *self-attention* multi-cabeza más una red *feed-forward*, con residuales y LayerNorm. Se apila con `nn.TransformerEncoder(layer, num_layers)`.

| Parámetro | Propósito |
|---|---|
| `d_model` | Dimensión de cada token. Las variables de entrada se proyectan a `d_model` con un `nn.Linear` (o, en PatchTST, se proyecta cada *patch* de 24 h). |
| `nhead` | Número de cabezas de atención; `d_model` debe ser divisible entre `nhead`. Cada cabeza trabaja en `d_model/nhead` dimensiones. |
| `dim_feedforward` | Ancho de la capa oculta de la FFN (por defecto 2048; para series suele usarse 2–4·`d_model`). |
| `dropout`, `activation` | Regularización y activación de la FFN (`relu` o `gelu`). |
| `batch_first` | Si es `True`, la entrada es `(B, S, d_model)`. |
| `norm_first` | Pre-LN (más estable al apilar muchas capas) contra Post-LN. |

En `forward`, `src_mask` (por ejemplo, una máscara causal triangular) y `src_key_padding_mask` restringen qué posiciones se atienden.

**Codificación posicional.** La atención es invariante a permutaciones, así que el orden temporal debe agregarse explícitamente: se suma a los tokens un vector que depende de la posición. Hay dos opciones:

1. **Sinusoidal fija**: $PE_{(p,2i)}=\sin(p/10000^{2i/d})$ y $PE_{(p,2i+1)}=\cos(p/10000^{2i/d})$, guardada con `register_buffer`.
2. **Aprendida**: un `nn.Parameter` de forma `(1, S, d_model)`, como en PatchTST.

En series de tiempo también se suman *embeddings* de calendario (hora, día) como codificación "temporal". iTransformer invierte la idea: cada **serie completa** es un token y la atención mezcla variables, no tiempos.

## NeuralForecast y StatsForecast (Nixtla)

**Formato de entrada.** Ambas usan un DataFrame "largo" con tres columnas obligatorias:

- `unique_id`: identificador de la serie (aquí, el cliente);
- `ds`: marca de tiempo (o un entero);
- `y`: el valor objetivo.

Las covariables son columnas adicionales. En NeuralForecast se declaran como `futr_exog_list` (conocidas a futuro, como el calendario; en `predict` se pasan en `futr_df`), `hist_exog_list` (solo pasadas) o `stat_exog_list` (estáticas, en `static_df`).

**Construcción y entrenamiento.**

- StatsForecast: `StatsForecast(models=[...], freq="h", n_jobs=-1)`. Ajusta un modelo **local** por serie, en paralelo por procesos.
- NeuralForecast: `NeuralForecast(models=[NHITS(h=24, input_size=168, ...)], freq="h")`. Entrena un modelo **global** con ventanas de todas las series. `fit(df, val_size=...)` reserva las últimas `val_size` observaciones de cada serie para validación y early stopping (`early_stop_patience_steps`, `val_check_steps`).

**`cross_validation`** hace el *backtesting* con origen móvil:

| Parámetro | Significado |
|---|---|
| `h` (StatsForecast) | Horizonte de cada ventana. En NeuralForecast va en el modelo. |
| `n_windows` | Número de orígenes (ventanas de prueba). Se toman al final de la serie: las ventanas cubren las últimas `h + step_size·(n_windows−1)` observaciones. |
| `step_size` | Distancia entre orígenes consecutivos (`24` = un origen por día). Si `step_size < h`, las ventanas se traslapan. |
| `refit` | `True` reentrena en cada ventana (caro). `False` entrena una sola vez con los datos previos a la primera ventana y en las siguientes solo usa los datos nuevos como entrada (en StatsForecast mediante `forward`, sin reajustar parámetros). Un entero `n` reentrena cada `n` ventanas. |
| `input_size` (StatsForecast) | Limita la historia que ve cada ajuste (ventana rodante en lugar de expandible). |
| `val_size`, `test_size` (NeuralForecast) | Separan la validación interna para early stopping y el bloque de prueba. |

La salida trae `unique_id`, `ds`, `cutoff` (el último instante observado en esa ventana), `y` y una columna por modelo. Con `level=[80]` trae además `-lo-80` y `-hi-80`.

## Chronos: `BaseChronosPipeline.predict_quantiles`

`BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base", device_map="mps", torch_dtype=torch.float32)` carga un modelo fundacional preentrenado en miles de series. Se usa *zero-shot*, sin entrenamiento.

- **Entrada** (`inputs`): un tensor 1D (una serie), un tensor 2D `(batch, longitud)` o una lista de tensores 1D de distinta longitud (se rellenan a la izquierda con NaN). Solo hay historia univariada, sin covariables. Cada contexto se escala internamente (*mean scaling*). Chronos-Bolt usa hasta 2048 pasos de contexto; si recibe más, se queda con los últimos.
- **`prediction_length`**: horizonte H. Bolt predice 64 pasos de forma directa; si se pide más, encadena pronósticos de 64 realimentando la mediana y emite una advertencia.
- **`quantile_levels`**: cuantiles pedidos (por defecto `[0.1, …, 0.9]`). Bolt devuelve directamente 9 cuantiles entrenados con pinball loss y aproxima los niveles intermedios. El Chronos original (T5) muestrea trayectorias y calcula los cuantiles empíricos.
- **Salida**: una tupla `(quantiles, mean)` con formas `(batch, prediction_length, n_quantiles)` y `(batch, prediction_length)`.

## Conceptos

**Modelo local contra modelo global.**

- Un modelo **local** ajusta parámetros propios por serie (ARIMA/ETS clásicos, o una red por cliente). Captura la idiosincrasia de cada serie, pero tiene pocos datos por modelo, no comparte patrones y no sirve para series nuevas (*cold start*). Además, mantener 370 modelos en producción es caro.
- Un modelo **global** comparte un solo conjunto de parámetros entre todas las series, normalizadas a una escala común. Ve muchos más ejemplos, aprende patrones comunes (perfil diario, feriados), regulariza y escala a series nuevas.
- Montero-Manso y Hyndman muestran que un global puede ser tan expresivo como los locales si tiene capacidad suficiente. Es la razón del éxito de N-BEATS, DeepAR y LightGBM global en competencias como M4/M5.

**Teacher forcing.**

- En modelos autorregresivos y en decoders seq2seq, durante el entrenamiento se alimenta el paso `t+1` con el **valor real** `y_t` en lugar de la predicción `ŷ_t`. Esto paraleliza el entrenamiento (toda la secuencia en una pasada) y lo estabiliza.
- El costo es el **sesgo de exposición**: en inferencia el modelo recibe sus propias predicciones, que nunca vio en entrenamiento, y los errores se acumulan a lo largo del horizonte.
- Las mitigaciones son *scheduled sampling* (usar la predicción con probabilidad creciente; aquí 0.5 en el seq2seq) o entrenar directamente multi-salida.

**RevIN (Reversible Instance Normalization; Kim et al., 2022).**

- Normaliza cada ventana de entrada con su propia media y desviación, $\tilde x = \gamma\,(x-\mu_{ventana})/\sigma_{ventana} + \beta$, con un afín aprendible.
- La red pronostica en esa escala y la salida se **desnormaliza** con los mismos $\mu,\sigma$ de la ventana: $\hat y = \sigma(\hat y' - \beta)/\gamma + \mu$.
- Ayuda ante cambios de nivel (*distribution shift*): si un cliente sube o baja su consumo después del periodo de entrenamiento, un z-score con estadísticos de entrenamiento entrega a la red valores fuera de su rango. Con RevIN la red siempre ve ventanas centradas y escaladas, y el nivel actual se restituye en la salida.
- PatchTST y la mayoría de los modelos de NeuralForecast (`scaler_type`) usan esta idea.

**Pinball loss y cuantiles.**

- Para el nivel $q\in(0,1)$: $L_q(y,\hat y)=\max\{q(y-\hat y),\,(q-1)(y-\hat y)\}$. Penaliza la subestimación con peso $q$ y la sobreestimación con peso $1-q$.
- Su minimizador en esperanza es exactamente el **cuantil $q$** de la distribución condicional, porque la derivada se anula cuando $P(Y\le\hat y)=q$.
- Con $q=0.5$ es la mitad del MAE (la mediana).
- Entrenar con varios $q$ da intervalos (P10–P90) sin suponer una distribución. Promediar la pinball sobre muchos cuantiles aproxima el CRPS.
- Un buen pronóstico probabilístico combina una pinball baja con una **cobertura** cercana a la nominal (80 % para P10–P90).

**MASE contra WAPE** (ambas libres de escala):

- **MASE** $=\frac{\text{MAE}}{\frac{1}{n-m}\sum_t |y_t-y_{t-m}|}$. Escala el error de cada serie con el MAE dentro de la muestra de un *seasonal naive* (aquí m = 24, en entrenamiento).
  - Es comparable entre series y cada serie pesa lo mismo en el promedio.
  - Está definida aunque haya ceros.
  - Tiene una lectura directa: < 1 es mejor que el naive estacional dentro de la muestra.
- **WAPE** $=\sum|e|/\sum|y|$ sobre todo el panel. Es un porcentaje de la energía total y está ponderado por volumen.
  - Los clientes grandes dominan, lo cual es relevante para el operador de la red, que se interesa por los kWh totales.
  - No se indefine con ceros individuales (solo si toda la demanda es cero), a diferencia del MAPE.
- Por eso un modelo puede mejorar en WAPE (acierta en los clientes grandes) y empeorar en MASE (falla en los pequeños), y viceversa.
