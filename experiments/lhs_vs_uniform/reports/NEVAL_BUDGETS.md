# Best score vs number of evaluations

_Generated 2026-08-19 10:04:30 UTC_

Budgets: `n_iter ∈ [3, 5, 10, 30]`. For each budget/method, as many seed repeats as fit in **5s** wall time (always **≥3**). Points: mean ± std; bands: 10th–90th percentiles. Winning hyperparameter combos are listed per repeat.

## `diabetes_ridge` — 1 tuned hparams

![neval_diabetes_ridge.png](figures_neval/neval_diabetes_ridge.png)

### Search space

```
alpha ~ loguniform(0.0001, 10000)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -3354.6±760 (n=495) | -3033.6±50.3 (n=486) |
| 5 | -3096.1±316 (n=297) | -3027.5±48.5 (n=294) |
| 10 | -3027.7±45.3 (n=149) | -3026±45 (n=149) |
| 30 | -3030.5±40.1 (n=50) | -3029.9±40.3 (n=50) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=495, wall=5.01s)

- Mean±std score: **-3354.6 ± 760**
- Best repeat: seed=209, score=-2908, `alpha=0.01907`

Winning-param summary across repeats:

- `alpha`: median=0.016612 [q10=0.0002702, q90=3.1252]

| seed | score | winning params |
|-----:|------:|----------------|
| 23 | -2943.2 | `alpha=0.0010694` |
| 35 | -2953.4 | `alpha=0.050535` |
| 51 | -2935 | `alpha=0.010946` |
| 63 | -2939.7 | `alpha=0.0037454` |
| 91 | -2949.5 | `alpha=0.018394` |
| 133 | -2963.7 | `alpha=0.077054` |
| 136 | -2945.1 | `alpha=0.001046` |
| 164 | -2962.3 | `alpha=0.00038399` |
| 209 | -2908 | `alpha=0.01907` |
| 222 | -2953.9 | `alpha=0.10415` |
| 231 | -2936 | `alpha=0.0017217` |
| 234 | -2945.8 | `alpha=0.01799` |
| 265 | -2966.8 | `alpha=0.033453` |
| 277 | -2967.4 | `alpha=0.0086812` |
| 287 | -2966.9 | `alpha=0.00011244` |
| 334 | -2965.4 | `alpha=0.00026528` |
| 336 | -2958.2 | `alpha=0.043172` |
| 348 | -2965.7 | `alpha=0.0049851` |
| 351 | -2966.3 | `alpha=0.0013236` |
| 451 | -2964.7 | `alpha=0.0033564` |
| 460 | -2958.3 | `alpha=0.00013959` |
| 465 | -2947.1 | `alpha=0.0021486` |
| 471 | -2950.2 | `alpha=0.0010605` |
| 490 | -2965.7 | `alpha=0.023379` |
| 491 | -2962.2 | `alpha=0.030372` |

_Showing 25/495 repeats (highest scores). Full list in JSON._

#### `lhs` @ n_iter=3 (n=486, wall=5.01s)

- Mean±std score: **-3033.6 ± 50.3**
- Best repeat: seed=209, score=-2908.4, `alpha=0.00037486`

Winning-param summary across repeats:

- `alpha`: median=0.0026214 [q10=0.00022455, q90=0.045253]

| seed | score | winning params |
|-----:|------:|----------------|
| 23 | -2943.6 | `alpha=0.0018208` |
| 35 | -2952.4 | `alpha=0.0058831` |
| 51 | -2930.4 | `alpha=0.0015212` |
| 63 | -2942.4 | `alpha=0.0010854` |
| 84 | -2958.2 | `alpha=0.03174` |
| 96 | -2944.3 | `alpha=0.00024501` |
| 131 | -2955.9 | `alpha=0.021975` |
| 136 | -2945.8 | `alpha=0.00035279` |
| 142 | -2957.6 | `alpha=0.029252` |
| 170 | -2938 | `alpha=0.00069312` |
| 174 | -2946.2 | `alpha=0.0286` |
| 201 | -2961.2 | `alpha=0.026366` |
| 209 | -2908.4 | `alpha=0.00037486` |
| 222 | -2936.8 | `alpha=0.0053843` |
| 231 | -2938 | `alpha=0.00061153` |
| 234 | -2940.9 | `alpha=0.0040182` |
| 243 | -2941.7 | `alpha=0.0061501` |
| 270 | -2937.9 | `alpha=0.0032201` |
| 321 | -2954.9 | `alpha=0.000445` |
| 336 | -2947.2 | `alpha=0.00032676` |
| 351 | -2946.2 | `alpha=0.023887` |
| 420 | -2952.7 | `alpha=0.0072739` |
| 460 | -2960.4 | `alpha=0.026807` |
| 465 | -2948.9 | `alpha=0.019159` |
| 471 | -2950.4 | `alpha=0.0023904` |

_Showing 25/486 repeats (highest scores). Full list in JSON._

### n_iter = 5

#### `uniform` @ n_iter=5 (n=297, wall=5.01s)

- Mean±std score: **-3096.1 ± 316**
- Best repeat: seed=209, score=-2906.4, `alpha=0.001289`

Winning-param summary across repeats:

- `alpha`: median=0.0070731 [q10=0.00024001, q90=0.1287]

| seed | score | winning params |
|-----:|------:|----------------|
| 23 | -2943.2 | `alpha=0.0008122` |
| 35 | -2953.4 | `alpha=0.050535` |
| 51 | -2933.3 | `alpha=0.0060388` |
| 63 | -2939.7 | `alpha=0.004009` |
| 84 | -2970.3 | `alpha=0.0066016` |
| 91 | -2949.2 | `alpha=0.021888` |
| 131 | -2969.4 | `alpha=0.0015572` |
| 133 | -2963.7 | `alpha=0.077054` |
| 136 | -2945.1 | `alpha=0.001046` |
| 140 | -2968.6 | `alpha=0.0061022` |
| 164 | -2962.3 | `alpha=0.00038399` |
| 168 | -2969.6 | `alpha=0.014514` |
| 170 | -2937.8 | `alpha=0.0030869` |
| 174 | -2936.9 | `alpha=0.0008002` |
| 209 | -2906.4 | `alpha=0.001289` |
| 222 | -2953.9 | `alpha=0.10415` |
| 231 | -2936 | `alpha=0.0017217` |
| 234 | -2945.8 | `alpha=0.01799` |
| 243 | -2949 | `alpha=0.020834` |
| 265 | -2966.8 | `alpha=0.033453` |
| 270 | -2937.3 | `alpha=0.0012404` |
| 273 | -2969.7 | `alpha=0.0034472` |
| 276 | -2964.8 | `alpha=0.0014084` |
| 277 | -2949.4 | `alpha=0.00020702` |
| 287 | -2966.9 | `alpha=0.00011244` |

_Showing 25/297 repeats (highest scores). Full list in JSON._

#### `lhs` @ n_iter=5 (n=294, wall=5.01s)

- Mean±std score: **-3027.5 ± 48.5**
- Best repeat: seed=209, score=-2908.9, `alpha=0.00022096`

Winning-param summary across repeats:

- `alpha`: median=0.0062204 [q10=0.00027461, q90=0.067878]

| seed | score | winning params |
|-----:|------:|----------------|
| 23 | -2943.3 | `alpha=0.00057038` |
| 35 | -2950.4 | `alpha=0.0294` |
| 51 | -2929.6 | `alpha=0.00051206` |
| 63 | -2941.1 | `alpha=0.0098177` |
| 84 | -2958.9 | `alpha=0.026822` |
| 91 | -2949.9 | `alpha=0.050568` |
| 96 | -2943.1 | `alpha=0.00103` |
| 131 | -2957.7 | `alpha=0.0095313` |
| 133 | -2964.8 | `alpha=0.01571` |
| 136 | -2945.8 | `alpha=0.0051573` |
| 140 | -2966.9 | `alpha=0.019935` |
| 164 | -2964.3 | `alpha=0.0018287` |
| 170 | -2938.5 | `alpha=0.00031951` |
| 174 | -2938.9 | `alpha=0.0029773` |
| 201 | -2957 | `alpha=0.00078495` |
| 209 | -2908.9 | `alpha=0.00022096` |
| 222 | -2938.9 | `alpha=0.0010931` |
| 231 | -2935.5 | `alpha=0.0057575` |
| 234 | -2938.5 | `alpha=0.00091711` |
| 243 | -2933.8 | `alpha=0.00019593` |
| 265 | -2965.4 | `alpha=0.0092074` |
| 270 | -2937.5 | `alpha=0.00013923` |
| 276 | -2963.7 | `alpha=0.0024696` |
| 277 | -2955.7 | `alpha=0.0019457` |
| 287 | -2963.9 | `alpha=0.0083341` |

_Showing 25/294 repeats (highest scores). Full list in JSON._

### n_iter = 10

#### `uniform` @ n_iter=10 (n=149, wall=5.02s)

- Mean±std score: **-3027.7 ± 45.3**
- Best repeat: seed=51, score=-2933, `alpha=0.0054158`

Winning-param summary across repeats:

- `alpha`: median=0.005704 [q10=0.0002314, q90=0.07662]

| seed | score | winning params |
|-----:|------:|----------------|
| 23 | -2943.2 | `alpha=0.0008122` |
| 24 | -2976.4 | `alpha=0.043868` |
| 35 | -2953.4 | `alpha=0.050535` |
| 51 | -2933 | `alpha=0.0054158` |
| 55 | -2973 | `alpha=0.00034856` |
| 57 | -2984.2 | `alpha=0.0017846` |
| 59 | -2973.2 | `alpha=0.003091` |
| 63 | -2939.7 | `alpha=0.004009` |
| 66 | -2978.6 | `alpha=0.0006647` |
| 72 | -2982 | `alpha=0.00011308` |
| 73 | -2977.9 | `alpha=0.0023259` |
| 83 | -2982.5 | `alpha=0.00022743` |
| 84 | -2961.2 | `alpha=0.018325` |
| 91 | -2949.2 | `alpha=0.021888` |
| 96 | -2943.7 | `alpha=0.0005515` |
| 106 | -2970.8 | `alpha=0.037424` |
| 110 | -2981.3 | `alpha=0.00035514` |
| 116 | -2974.3 | `alpha=0.0023931` |
| 123 | -2982 | `alpha=0.00026951` |
| 129 | -2973.1 | `alpha=0.072355` |
| 131 | -2969.4 | `alpha=0.0015572` |
| 133 | -2963.7 | `alpha=0.077054` |
| 136 | -2944.8 | `alpha=0.0025192` |
| 140 | -2968.6 | `alpha=0.0061022` |
| 142 | -2953.3 | `alpha=0.062472` |

_Showing 25/149 repeats (highest scores). Full list in JSON._

#### `lhs` @ n_iter=10 (n=149, wall=5.03s)

- Mean±std score: **-3026 ± 45**
- Best repeat: seed=51, score=-2929.4, `alpha=0.00013228`

Winning-param summary across repeats:

- `alpha`: median=0.013876 [q10=0.00031347, q90=0.092]

| seed | score | winning params |
|-----:|------:|----------------|
| 23 | -2943.4 | `alpha=0.0015069` |
| 24 | -2979.4 | `alpha=0.0084895` |
| 35 | -2949.9 | `alpha=0.015949` |
| 51 | -2929.4 | `alpha=0.00013228` |
| 55 | -2972.6 | `alpha=0.0020524` |
| 57 | -2983 | `alpha=0.00044248` |
| 59 | -2973 | `alpha=0.0042477` |
| 63 | -2940.7 | `alpha=0.0085337` |
| 66 | -2978.6 | `alpha=0.00031504` |
| 72 | -2982 | `alpha=0.00012166` |
| 73 | -2979.2 | `alpha=0.00064537` |
| 83 | -2982.5 | `alpha=0.00026439` |
| 84 | -2967.7 | `alpha=0.0084874` |
| 91 | -2949 | `alpha=0.025404` |
| 96 | -2942.8 | `alpha=0.0026131` |
| 106 | -2975.3 | `alpha=0.0044321` |
| 110 | -2981.3 | `alpha=0.00045728` |
| 116 | -2974.2 | `alpha=0.0028399` |
| 123 | -2982.1 | `alpha=0.00039935` |
| 129 | -2972.7 | `alpha=0.036451` |
| 131 | -2957.1 | `alpha=0.048743` |
| 133 | -2965.1 | `alpha=0.0064191` |
| 136 | -2945.1 | `alpha=0.00093183` |
| 140 | -2967.2 | `alpha=0.056209` |
| 142 | -2962.3 | `alpha=0.019602` |

_Showing 25/149 repeats (highest scores). Full list in JSON._

### n_iter = 30

#### `uniform` @ n_iter=30 (n=50, wall=5.03s)

- Mean±std score: **-3030.5 ± 40.1**
- Best repeat: seed=23, score=-2943.2, `alpha=0.0008122`

Winning-param summary across repeats:

- `alpha`: median=0.01889 [q10=0.00020801, q90=0.069079]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -2991.7 | `alpha=0.0025424` |
| 2 | -3016.1 | `alpha=0.012385` |
| 5 | -3021.6 | `alpha=0.0065849` |
| 7 | -2988.8 | `alpha=0.026562` |
| 9 | -2989.2 | `alpha=0.033483` |
| 11 | -3003.5 | `alpha=0.015939` |
| 12 | -3006.4 | `alpha=0.011754` |
| 14 | -2982.9 | `alpha=0.0018447` |
| 18 | -3012.7 | `alpha=0.0011093` |
| 20 | -3017.7 | `alpha=0.043696` |
| 21 | -3013.7 | `alpha=0.00010049` |
| 22 | -2993.5 | `alpha=0.00025184` |
| 23 | -2943.2 | `alpha=0.0008122` |
| 24 | -2976.4 | `alpha=0.043868` |
| 31 | -2999.4 | `alpha=0.019005` |
| 35 | -2950.3 | `alpha=0.012051` |
| 36 | -3012.7 | `alpha=0.042557` |
| 37 | -2998.7 | `alpha=0.029918` |
| 39 | -3016.7 | `alpha=0.028255` |
| 40 | -2987.3 | `alpha=0.00021809` |
| 43 | -3017.7 | `alpha=0.0001078` |
| 44 | -2998.6 | `alpha=0.0027948` |
| 46 | -2997.6 | `alpha=0.00011391` |
| 47 | -3005.3 | `alpha=0.0027159` |
| 48 | -3021.2 | `alpha=0.027501` |

_Showing 25/50 repeats (highest scores). Full list in JSON._

#### `lhs` @ n_iter=30 (n=50, wall=5.04s)

- Mean±std score: **-3029.9 ± 40.3**
- Best repeat: seed=23, score=-2943.2, `alpha=0.0007152`

Winning-param summary across repeats:

- `alpha`: median=0.030911 [q10=0.00015895, q90=0.079727]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -2991.2 | `alpha=0.049677` |
| 2 | -3016 | `alpha=0.0096783` |
| 5 | -3020.6 | `alpha=0.0029581` |
| 7 | -2986.8 | `alpha=0.079114` |
| 9 | -2989.5 | `alpha=0.046249` |
| 11 | -3001.2 | `alpha=0.038409` |
| 12 | -3006.1 | `alpha=0.029728` |
| 14 | -2982.8 | `alpha=0.0028845` |
| 18 | -3012.5 | `alpha=0.0023282` |
| 20 | -3015.9 | `alpha=0.071462` |
| 21 | -3013.7 | `alpha=0.00015799` |
| 22 | -2993.4 | `alpha=0.00015906` |
| 23 | -2943.2 | `alpha=0.0007152` |
| 24 | -2976.3 | `alpha=0.035442` |
| 31 | -2999.3 | `alpha=0.0085333` |
| 35 | -2949.9 | `alpha=0.019891` |
| 36 | -3012 | `alpha=0.050475` |
| 37 | -2999.5 | `alpha=0.019131` |
| 39 | -3015.3 | `alpha=0.053881` |
| 40 | -2987.3 | `alpha=0.00015697` |
| 43 | -3017.7 | `alpha=0.00010111` |
| 44 | -2998.6 | `alpha=0.0023498` |
| 46 | -2997.6 | `alpha=0.00016969` |
| 47 | -3005.1 | `alpha=0.0018952` |
| 48 | -3021 | `alpha=0.032093` |

_Showing 25/50 repeats (highest scores). Full list in JSON._

## `diabetes_hgb` — 4 tuned hparams

![neval_diabetes_hgb.png](figures_neval/neval_diabetes_hgb.png)

### Search space

```
learning_rate ~ loguniform(0.001, 1)
max_depth ~ randint(1, 16)
min_samples_leaf ~ randint(1, 80)
l2_regularization ~ loguniform(1e-08, 100)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -3548.4±401 (n=12) | -3436.5±245 (n=11) |
| 5 | -3396±181 (n=8) | -3353.8±85.8 (n=6) |
| 10 | -3332.1±81.8 (n=4) | -3294.1±111 (n=4) |
| 30 | -3242.4±25.4 (n=3) | -3274.3±65.5 (n=3) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=12, wall=5.29s)

- Mean±std score: **-3548.4 ± 401**
- Best repeat: seed=7, score=-3187.1, `l2_regularization=1.7868e-06, learning_rate=0.075039, max_depth=15, min_samples_leaf=63`

Winning-param summary across repeats:

- `learning_rate`: median=0.070712 [q10=0.035367, q90=0.25]
- `max_depth`: median=11 [q10=2, q90=15]
- `min_samples_leaf`: median=49 [q10=17, q90=65.9]
- `l2_regularization`: median=0.0013851 [q10=8.7667e-08, q90=3.1201]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3367.7 | `l2_regularization=1.0651e-08, learning_rate=0.042744, max_depth=15, min_samples_leaf=66` |
| 1 | -3543.1 | `l2_regularization=0.0024068, learning_rate=0.044543, max_depth=1, min_samples_leaf=61` |
| 2 | -3487 | `l2_regularization=3.5601e-08, learning_rate=0.06314, max_depth=12, min_samples_leaf=16` |
| 3 | -3443.7 | `l2_regularization=0.0014703, learning_rate=0.15986, max_depth=2, min_samples_leaf=32` |
| 4 | -3218.2 | `l2_regularization=5.5626e-07, learning_rate=0.066385, max_depth=7, min_samples_leaf=65` |
| 5 | -3748.8 | `l2_regularization=7.2113e-06, learning_rate=0.26002, max_depth=13, min_samples_leaf=42` |
| 6 | -3298.9 | `l2_regularization=3.1762, learning_rate=0.10959, max_depth=2, min_samples_leaf=5` |
| 7 | -3187.1 | `l2_regularization=1.7868e-06, learning_rate=0.075039, max_depth=15, min_samples_leaf=63` |
| 8 | -4004.8 | `l2_regularization=0.76823, learning_rate=0.0095701, max_depth=16, min_samples_leaf=26` |
| 9 | -3398.9 | `l2_regularization=15.219, learning_rate=0.14068, max_depth=15, min_samples_leaf=69` |
| 10 | -3294.2 | `l2_regularization=2.615, learning_rate=0.034547, max_depth=3, min_samples_leaf=56` |
| 11 | -4588.5 | `l2_regularization=0.0012999, learning_rate=0.69982, max_depth=10, min_samples_leaf=30` |

#### `lhs` @ n_iter=3 (n=11, wall=5.08s)

- Mean±std score: **-3436.5 ± 245**
- Best repeat: seed=4, score=-3206.9, `l2_regularization=0.11064, learning_rate=0.082296, max_depth=11, min_samples_leaf=77`

Winning-param summary across repeats:

- `learning_rate`: median=0.10138 [q10=0.018631, q90=0.62496]
- `max_depth`: median=11 [q10=6, q90=14]
- `min_samples_leaf`: median=64 [q10=33, q90=77]
- `l2_regularization`: median=4.184e-05 [q10=4.662e-07, q90=0.11595]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3313.5 | `l2_regularization=0.11595, learning_rate=0.14027, max_depth=14, min_samples_leaf=61` |
| 1 | -3967.1 | `l2_regularization=0.0047946, learning_rate=0.62496, max_depth=6, min_samples_leaf=64` |
| 2 | -3583.9 | `l2_regularization=3.1938e-05, learning_rate=0.01663, max_depth=14, min_samples_leaf=50` |
| 3 | -3693.4 | `l2_regularization=5.9798e-08, learning_rate=0.65675, max_depth=13, min_samples_leaf=79` |
| 4 | -3206.9 | `l2_regularization=0.11064, learning_rate=0.082296, max_depth=11, min_samples_leaf=77` |
| 5 | -3313.8 | `l2_regularization=0.018917, learning_rate=0.10138, max_depth=11, min_samples_leaf=60` |
| 6 | -3208.9 | `l2_regularization=3.3849e-05, learning_rate=0.089686, max_depth=8, min_samples_leaf=70` |
| 7 | -3224.5 | `l2_regularization=7.8494e-06, learning_rate=0.20239, max_depth=6, min_samples_leaf=69` |
| 8 | -3498.1 | `l2_regularization=4.662e-07, learning_rate=0.061798, max_depth=9, min_samples_leaf=23` |
| 9 | -3542.2 | `l2_regularization=0.17637, learning_rate=0.018631, max_depth=6, min_samples_leaf=33` |
| 10 | -3249.3 | `l2_regularization=4.184e-05, learning_rate=0.27408, max_depth=12, min_samples_leaf=64` |

### n_iter = 5

#### `uniform` @ n_iter=5 (n=8, wall=5.51s)

- Mean±std score: **-3396 ± 181**
- Best repeat: seed=7, score=-3187.1, `l2_regularization=1.7868e-06, learning_rate=0.075039, max_depth=15, min_samples_leaf=63`

Winning-param summary across repeats:

- `learning_rate`: median=0.070712 [q10=0.02008, q90=0.22393]
- `max_depth`: median=4.5 [q10=1, q90=15.3]
- `min_samples_leaf`: median=60 [q10=23.9, q90=67.1]
- `l2_regularization`: median=0.0019385 [q10=5.665e-07, q90=2.8911]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3310.6 | `l2_regularization=5.7089e-07, learning_rate=0.37343, max_depth=1, min_samples_leaf=59` |
| 1 | -3543.1 | `l2_regularization=0.0024068, learning_rate=0.044543, max_depth=1, min_samples_leaf=61` |
| 2 | -3434.5 | `l2_regularization=0.021471, learning_rate=0.019856, max_depth=11, min_samples_leaf=34` |
| 3 | -3443.7 | `l2_regularization=0.0014703, learning_rate=0.15986, max_depth=2, min_samples_leaf=32` |
| 4 | -3218.2 | `l2_regularization=5.5626e-07, learning_rate=0.066385, max_depth=7, min_samples_leaf=65` |
| 5 | -3731.7 | `l2_regularization=2.7689, learning_rate=0.020176, max_depth=16, min_samples_leaf=72` |
| 6 | -3298.9 | `l2_regularization=3.1762, learning_rate=0.10959, max_depth=2, min_samples_leaf=5` |
| 7 | -3187.1 | `l2_regularization=1.7868e-06, learning_rate=0.075039, max_depth=15, min_samples_leaf=63` |

#### `lhs` @ n_iter=5 (n=6, wall=5.05s)

- Mean±std score: **-3353.8 ± 85.8**
- Best repeat: seed=0, score=-3237.2, `l2_regularization=3.2988e-08, learning_rate=0.062097, max_depth=2, min_samples_leaf=31`

Winning-param summary across repeats:

- `learning_rate`: median=0.061813 [q10=0.032854, q90=0.075411]
- `max_depth`: median=7 [q10=3.5, q90=14.5]
- `min_samples_leaf`: median=53.5 [q10=30.5, q90=76]
- `l2_regularization`: median=4.5398e-05 [q10=6.4972e-07, q90=3.0047]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3237.2 | `l2_regularization=3.2988e-08, learning_rate=0.062097, max_depth=2, min_samples_leaf=31` |
| 1 | -3369.6 | `l2_regularization=1.2597e-05, learning_rate=0.061529, max_depth=5, min_samples_leaf=49` |
| 2 | -3255.2 | `l2_regularization=1.2664e-06, learning_rate=0.085611, max_depth=9, min_samples_leaf=78` |
| 3 | -3419.2 | `l2_regularization=5.0174, learning_rate=0.06521, max_depth=13, min_samples_leaf=30` |
| 4 | -3426.5 | `l2_regularization=7.8198e-05, learning_rate=0.029615, max_depth=5, min_samples_leaf=74` |
| 5 | -3414.7 | `l2_regularization=0.99198, learning_rate=0.036092, max_depth=16, min_samples_leaf=58` |

### n_iter = 10

#### `uniform` @ n_iter=10 (n=4, wall=5.13s)

- Mean±std score: **-3332.1 ± 81.8**
- Best repeat: seed=1, score=-3256.2, `l2_regularization=64.176, learning_rate=0.17828, max_depth=5, min_samples_leaf=39`

Winning-param summary across repeats:

- `learning_rate`: median=0.13689 [q10=0.059, q90=0.17275]
- `max_depth`: median=7.5 [q10=2.9, q90=10.7]
- `min_samples_leaf`: median=47.5 [q10=34.1, q90=65.1]
- `l2_regularization`: median=0.0008556 [q10=0.00012651, q90=44.924]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3289.5 | `l2_regularization=7.7484e-05, learning_rate=0.11393, max_depth=11, min_samples_leaf=56` |
| 1 | -3256.2 | `l2_regularization=64.176, learning_rate=0.17828, max_depth=5, min_samples_leaf=39` |
| 2 | -3339.1 | `l2_regularization=0.00024091, learning_rate=0.03546, max_depth=10, min_samples_leaf=69` |
| 3 | -3443.7 | `l2_regularization=0.0014703, learning_rate=0.15986, max_depth=2, min_samples_leaf=32` |

#### `lhs` @ n_iter=10 (n=4, wall=5.40s)

- Mean±std score: **-3294.1 ± 111**
- Best repeat: seed=3, score=-3163.6, `l2_regularization=1.702e-07, learning_rate=0.23423, max_depth=1, min_samples_leaf=11`

Winning-param summary across repeats:

- `learning_rate`: median=0.12492 [q10=0.12451, q90=0.20145]
- `max_depth`: median=8 [q10=1.9, q90=13.4]
- `min_samples_leaf`: median=60.5 [q10=24.5, q90=74.8]
- `l2_regularization`: median=0.001696 [q10=1.5468e-07, q90=12.715]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3299.4 | `l2_regularization=18.163, learning_rate=0.12489, max_depth=14, min_samples_leaf=56` |
| 1 | -3435.2 | `l2_regularization=1.4803e-07, learning_rate=0.12496, max_depth=4, min_samples_leaf=79` |
| 2 | -3278.3 | `l2_regularization=0.0033918, learning_rate=0.12435, max_depth=12, min_samples_leaf=65` |
| 3 | -3163.6 | `l2_regularization=1.702e-07, learning_rate=0.23423, max_depth=1, min_samples_leaf=11` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=3, wall=11.96s)

- Mean±std score: **-3242.4 ± 25.4**
- Best repeat: seed=2, score=-3213.1, `l2_regularization=0.014005, learning_rate=0.24058, max_depth=1, min_samples_leaf=79`

Winning-param summary across repeats:

- `learning_rate`: median=0.17828 [q10=0.07712, q90=0.22812]
- `max_depth`: median=5 [q10=1.8, q90=5.8]
- `min_samples_leaf`: median=48 [q10=40.8, q90=72.8]
- `l2_regularization`: median=0.014005 [q10=0.0028201, q90=51.344]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3257.9 | `l2_regularization=2.3939e-05, learning_rate=0.051831, max_depth=6, min_samples_leaf=48` |
| 1 | -3256.2 | `l2_regularization=64.176, learning_rate=0.17828, max_depth=5, min_samples_leaf=39` |
| 2 | -3213.1 | `l2_regularization=0.014005, learning_rate=0.24058, max_depth=1, min_samples_leaf=79` |

#### `lhs` @ n_iter=30 (n=3, wall=13.15s)

- Mean±std score: **-3274.3 ± 65.5**
- Best repeat: seed=2, score=-3198.9, `l2_regularization=0.00028286, learning_rate=0.071855, max_depth=12, min_samples_leaf=72`

Winning-param summary across repeats:

- `learning_rate`: median=0.084939 [q10=0.074471, q90=0.1137]
- `max_depth`: median=12 [q10=6.4, q90=14.4]
- `min_samples_leaf`: median=59 [q10=39.8, q90=69.4]
- `l2_regularization`: median=0.15305 [q10=0.030836, q90=39.672]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -3317.5 | `l2_regularization=0.15305, learning_rate=0.1209, max_depth=15, min_samples_leaf=59` |
| 1 | -3306.5 | `l2_regularization=49.552, learning_rate=0.084939, max_depth=5, min_samples_leaf=35` |
| 2 | -3198.9 | `l2_regularization=0.00028286, learning_rate=0.071855, max_depth=12, min_samples_leaf=72` |

## `breast_cancer_logreg` — 2 tuned hparams

![neval_breast_cancer_logreg.png](figures_neval/neval_breast_cancer_logreg.png)

### Search space

```
C ~ loguniform(0.0001, 10000)
l1_ratio ~ uniform(0, 1)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -0.2405±0.0026 (n=17) | -0.24047±0.00262 (n=17) |
| 5 | -0.24006±0.00229 (n=11) | -0.24006±0.00229 (n=11) |
| 10 | -0.23952±0.0016 (n=5) | -0.23925±0.00158 (n=6) |
| 30 | -0.23992±0.00209 (n=3) | -0.23992±0.00209 (n=3) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=17, wall=5.22s)

- Mean±std score: **-0.2405 ± 0.0026**
- Best repeat: seed=15, score=-0.23779, `C=34.829, l1_ratio=0.81582`

Winning-param summary across repeats:

- `C`: median=307.74 [q10=4.389, q90=5250.2]
- `l1_ratio`: median=0.28682 [q10=0.059977, q90=0.85459]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=320.75, l1_ratio=0.91276` |
| 1 | -0.23884 | `C=1.2433, l1_ratio=0.95046` |
| 2 | -0.24234 | `C=326.44, l1_ratio=0.091916` |
| 3 | -0.23928 | `C=257.16, l1_ratio=0.58216` |
| 4 | -0.23854 | `C=6455.8, l1_ratio=0.080836` |
| 5 | -0.2379 | `C=275.44, l1_ratio=0.80794` |
| 6 | -0.24163 | `C=7935.2, l1_ratio=0.63276` |
| 7 | -0.2449 | `C=160.5, l1_ratio=0.22521` |
| 8 | -0.24204 | `C=910.27, l1_ratio=0.39108` |
| 9 | -0.23853 | `C=916.21, l1_ratio=0.28682` |
| 10 | -0.23805 | `C=4446.5, l1_ratio=0.20768` |
| 11 | -0.24589 | `C=6.4861, l1_ratio=0.028689` |
| 12 | -0.23894 | `C=0.062967, l1_ratio=0.23054` |
| 13 | -0.24325 | `C=307.74, l1_ratio=0.26145` |
| 14 | -0.24278 | `C=444.49, l1_ratio=0.36095` |
| 15 | -0.23779 | `C=34.829, l1_ratio=0.81582` |
| 16 | -0.23912 | `C=9.3772, l1_ratio=0.021655` |

#### `lhs` @ n_iter=3 (n=17, wall=5.13s)

- Mean±std score: **-0.24047 ± 0.00262**
- Best repeat: seed=15, score=-0.23779, `C=76.445, l1_ratio=0.48199`

Winning-param summary across repeats:

- `C`: median=316.97 [q10=41.557, q90=5405.9]
- `l1_ratio`: median=0.50059 [q10=0.15028, q90=0.83319]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=53.122, l1_ratio=0.16391` |
| 1 | -0.23879 | `C=24.211, l1_ratio=0.85005` |
| 2 | -0.24234 | `C=4072.2, l1_ratio=0.35043` |
| 3 | -0.23928 | `C=58.371, l1_ratio=0.69992` |
| 4 | -0.23854 | `C=12.814, l1_ratio=0.3221` |
| 5 | -0.2379 | `C=84.003, l1_ratio=0.62768` |
| 6 | -0.24163 | `C=7480.4, l1_ratio=0.12983` |
| 7 | -0.2449 | `C=141.2, l1_ratio=0.69893` |
| 8 | -0.24204 | `C=1803.4, l1_ratio=0.57625` |
| 9 | -0.23853 | `C=537.62, l1_ratio=0.0028695` |
| 10 | -0.23805 | `C=316.97, l1_ratio=0.68928` |
| 11 | -0.24589 | `C=3002.3, l1_ratio=0.9775` |
| 12 | -0.23855 | `C=116.16, l1_ratio=0.37096` |
| 13 | -0.24325 | `C=7406.5, l1_ratio=0.50059` |
| 14 | -0.24278 | `C=1435.8, l1_ratio=0.3767` |
| 15 | -0.23779 | `C=76.445, l1_ratio=0.48199` |
| 16 | -0.23912 | `C=326.4, l1_ratio=0.82195` |

### n_iter = 5

#### `uniform` @ n_iter=5 (n=11, wall=5.49s)

- Mean±std score: **-0.24006 ± 0.00229**
- Best repeat: seed=5, score=-0.2379, `C=275.44, l1_ratio=0.80794`

Winning-param summary across repeats:

- `C`: median=418.43 [q10=160.5, q90=6455.8]
- `l1_ratio`: median=0.28682 [q10=0.091916, q90=0.80794]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=320.75, l1_ratio=0.91276` |
| 1 | -0.23878 | `C=418.43, l1_ratio=0.4092` |
| 2 | -0.24234 | `C=326.44, l1_ratio=0.091916` |
| 3 | -0.23928 | `C=75.269, l1_ratio=0.11367` |
| 4 | -0.23854 | `C=6455.8, l1_ratio=0.080836` |
| 5 | -0.2379 | `C=275.44, l1_ratio=0.80794` |
| 6 | -0.24163 | `C=7935.2, l1_ratio=0.63276` |
| 7 | -0.2449 | `C=160.5, l1_ratio=0.22521` |
| 8 | -0.24204 | `C=910.27, l1_ratio=0.39108` |
| 9 | -0.23853 | `C=916.21, l1_ratio=0.28682` |
| 10 | -0.23805 | `C=4446.5, l1_ratio=0.20768` |

#### `lhs` @ n_iter=5 (n=11, wall=5.42s)

- Mean±std score: **-0.24006 ± 0.00229**
- Best repeat: seed=5, score=-0.2379, `C=568.3, l1_ratio=0.57661`

Winning-param summary across repeats:

- `C`: median=1967.7 [q10=652.47, q90=5103.2]
- `l1_ratio`: median=0.22263 [q10=0.018656, q90=0.60172]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=652.47, l1_ratio=0.023856` |
| 1 | -0.23878 | `C=4713.7, l1_ratio=0.018656` |
| 2 | -0.24234 | `C=1109.7, l1_ratio=0.085005` |
| 3 | -0.23928 | `C=5103.2, l1_ratio=0.8693` |
| 4 | -0.23854 | `C=6210.5, l1_ratio=0.22263` |
| 5 | -0.2379 | `C=568.3, l1_ratio=0.57661` |
| 6 | -0.24163 | `C=2276, l1_ratio=0.21177` |
| 7 | -0.2449 | `C=1967.7, l1_ratio=0.37369` |
| 8 | -0.24204 | `C=4629.8, l1_ratio=0.5346` |
| 9 | -0.23853 | `C=1730.9, l1_ratio=0.60172` |
| 10 | -0.23805 | `C=1260.7, l1_ratio=0.013569` |

### n_iter = 10

#### `uniform` @ n_iter=10 (n=5, wall=5.24s)

- Mean±std score: **-0.23952 ± 0.0016**
- Best repeat: seed=4, score=-0.23854, `C=6455.8, l1_ratio=0.080836`

Winning-param summary across repeats:

- `C`: median=418.43 [q10=179.71, q90=6069]
- `l1_ratio`: median=0.11367 [q10=0.033978, q90=0.57352]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=336.38, l1_ratio=0.0027385` |
| 1 | -0.23878 | `C=418.43, l1_ratio=0.4092` |
| 2 | -0.24234 | `C=5488.9, l1_ratio=0.68306` |
| 3 | -0.23928 | `C=75.269, l1_ratio=0.11367` |
| 4 | -0.23854 | `C=6455.8, l1_ratio=0.080836` |

#### `lhs` @ n_iter=10 (n=6, wall=6.03s)

- Mean±std score: **-0.23925 ± 0.00158**
- Best repeat: seed=5, score=-0.2379, `C=7976, l1_ratio=0.77325`

Winning-param summary across repeats:

- `C`: median=4807.7 [q10=936.8, q90=7700.3]
- `l1_ratio`: median=0.21606 [q10=0.056779, q90=0.61262]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=7232.3, l1_ratio=0.30336` |
| 1 | -0.23878 | `C=2383, l1_ratio=0.10224` |
| 2 | -0.24234 | `C=7424.5, l1_ratio=0.45199` |
| 3 | -0.23928 | `C=624.6, l1_ratio=0.12877` |
| 4 | -0.23854 | `C=1249, l1_ratio=0.011317` |
| 5 | -0.2379 | `C=7976, l1_ratio=0.77325` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=3, wall=9.39s)

- Mean±std score: **-0.23992 ± 0.00209**
- Best repeat: seed=0, score=-0.23865, `C=336.38, l1_ratio=0.0027385`

Winning-param summary across repeats:

- `C`: median=4934.7 [q10=1256, q90=5378.1]
- `l1_ratio`: median=0.68306 [q10=0.1388, q90=0.71644]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=336.38, l1_ratio=0.0027385` |
| 1 | -0.23878 | `C=4934.7, l1_ratio=0.72479` |
| 2 | -0.24234 | `C=5488.9, l1_ratio=0.68306` |

#### `lhs` @ n_iter=30 (n=3, wall=9.05s)

- Mean±std score: **-0.23992 ± 0.00209**
- Best repeat: seed=0, score=-0.23865, `C=6343.4, l1_ratio=0.30808`

Winning-param summary across repeats:

- `C`: median=4608.2 [q10=4510.9, q90=5996.4]
- `l1_ratio`: median=0.18838 [q10=0.11803, q90=0.28414]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.23865 | `C=6343.4, l1_ratio=0.30808` |
| 1 | -0.23878 | `C=4486.5, l1_ratio=0.10044` |
| 2 | -0.24234 | `C=4608.2, l1_ratio=0.18838` |

## `breast_cancer_hgb` — 4 tuned hparams

![neval_breast_cancer_hgb.png](figures_neval/neval_breast_cancer_hgb.png)

### Search space

```
learning_rate ~ loguniform(0.001, 1)
max_depth ~ randint(1, 16)
min_samples_leaf ~ randint(1, 80)
l2_regularization ~ loguniform(1e-08, 100)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -0.13515±0.0356 (n=10) | -0.13123±0.0464 (n=11) |
| 5 | -0.12473±0.0282 (n=7) | -0.11408±0.0187 (n=7) |
| 10 | -0.11025±0.0236 (n=4) | -0.10225±0.0139 (n=4) |
| 30 | -0.099571±0.0118 (n=3) | -0.098318±0.0153 (n=3) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=10, wall=5.01s)

- Mean±std score: **-0.13515 ± 0.0356**
- Best repeat: seed=4, score=-0.10111, `l2_regularization=5.5626e-07, learning_rate=0.066385, max_depth=7, min_samples_leaf=65`

Winning-param summary across repeats:

- `learning_rate`: median=0.12513 [q10=0.06128, q90=0.28848]
- `max_depth`: median=9.5 [q10=1.9, q90=15]
- `min_samples_leaf`: median=45.5 [q10=14.9, q90=65.4]
- `l2_regularization`: median=0.00076184 [q10=5.0419e-07, q90=4.3805]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.12151 | `l2_regularization=0.19723, learning_rate=0.2753, max_depth=15, min_samples_leaf=49` |
| 1 | -0.16122 | `l2_regularization=0.0024068, learning_rate=0.044543, max_depth=1, min_samples_leaf=61` |
| 2 | -0.11817 | `l2_regularization=3.5601e-08, learning_rate=0.06314, max_depth=12, min_samples_leaf=16` |
| 3 | -0.10433 | `l2_regularization=0.0014703, learning_rate=0.15986, max_depth=2, min_samples_leaf=32` |
| 4 | -0.10111 | `l2_regularization=5.5626e-07, learning_rate=0.066385, max_depth=7, min_samples_leaf=65` |
| 5 | -0.16747 | `l2_regularization=7.2113e-06, learning_rate=0.26002, max_depth=13, min_samples_leaf=42` |
| 6 | -0.12042 | `l2_regularization=3.1762, learning_rate=0.10959, max_depth=2, min_samples_leaf=5` |
| 7 | -0.10737 | `l2_regularization=1.7868e-06, learning_rate=0.075039, max_depth=15, min_samples_leaf=63` |
| 8 | -0.21315 | `l2_regularization=5.3394e-05, learning_rate=0.40709, max_depth=7, min_samples_leaf=36` |
| 9 | -0.1368 | `l2_regularization=15.219, learning_rate=0.14068, max_depth=15, min_samples_leaf=69` |

#### `lhs` @ n_iter=3 (n=11, wall=5.45s)

- Mean±std score: **-0.13123 ± 0.0464**
- Best repeat: seed=0, score=-0.095835, `l2_regularization=0.11595, learning_rate=0.14027, max_depth=14, min_samples_leaf=61`

Winning-param summary across repeats:

- `learning_rate`: median=0.12154 [q10=0.061798, q90=0.62496]
- `max_depth`: median=9 [q10=6, q90=13]
- `min_samples_leaf`: median=64 [q10=23, q90=77]
- `l2_regularization`: median=0.0047946 [q10=4.662e-07, q90=0.17637]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.095835 | `l2_regularization=0.11595, learning_rate=0.14027, max_depth=14, min_samples_leaf=61` |
| 1 | -0.1645 | `l2_regularization=0.0047946, learning_rate=0.62496, max_depth=6, min_samples_leaf=64` |
| 2 | -0.10833 | `l2_regularization=1.2118, learning_rate=0.12154, max_depth=6, min_samples_leaf=11` |
| 3 | -0.18296 | `l2_regularization=5.9798e-08, learning_rate=0.65675, max_depth=13, min_samples_leaf=79` |
| 4 | -0.09675 | `l2_regularization=0.11064, learning_rate=0.082296, max_depth=11, min_samples_leaf=77` |
| 5 | -0.11154 | `l2_regularization=0.018917, learning_rate=0.10138, max_depth=11, min_samples_leaf=60` |
| 6 | -0.10164 | `l2_regularization=3.3849e-05, learning_rate=0.089686, max_depth=8, min_samples_leaf=70` |
| 7 | -0.10243 | `l2_regularization=7.8494e-06, learning_rate=0.20239, max_depth=6, min_samples_leaf=69` |
| 8 | -0.1239 | `l2_regularization=4.662e-07, learning_rate=0.061798, max_depth=9, min_samples_leaf=23` |
| 9 | -0.24222 | `l2_regularization=0.17637, learning_rate=0.018631, max_depth=6, min_samples_leaf=33` |
| 10 | -0.11345 | `l2_regularization=4.184e-05, learning_rate=0.27408, max_depth=12, min_samples_leaf=64` |

### n_iter = 5

#### `uniform` @ n_iter=5 (n=7, wall=5.32s)

- Mean±std score: **-0.12473 ± 0.0282**
- Best repeat: seed=0, score=-0.10037, `l2_regularization=5.7089e-07, learning_rate=0.37343, max_depth=1, min_samples_leaf=59`

Winning-param summary across repeats:

- `learning_rate`: median=0.10959 [q10=0.055701, q90=0.30539]
- `max_depth`: median=2 [q10=1, q90=12.4]
- `min_samples_leaf`: median=42 [q10=11.6, q90=62.6]
- `l2_regularization`: median=7.2113e-06 [q10=3.48e-07, q90=1.2719]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.10037 | `l2_regularization=5.7089e-07, learning_rate=0.37343, max_depth=1, min_samples_leaf=59` |
| 1 | -0.16122 | `l2_regularization=0.0024068, learning_rate=0.044543, max_depth=1, min_samples_leaf=61` |
| 2 | -0.11817 | `l2_regularization=3.5601e-08, learning_rate=0.06314, max_depth=12, min_samples_leaf=16` |
| 3 | -0.10433 | `l2_regularization=0.0014703, learning_rate=0.15986, max_depth=2, min_samples_leaf=32` |
| 4 | -0.10111 | `l2_regularization=5.5626e-07, learning_rate=0.066385, max_depth=7, min_samples_leaf=65` |
| 5 | -0.16747 | `l2_regularization=7.2113e-06, learning_rate=0.26002, max_depth=13, min_samples_leaf=42` |
| 6 | -0.12042 | `l2_regularization=3.1762, learning_rate=0.10959, max_depth=2, min_samples_leaf=5` |

#### `lhs` @ n_iter=5 (n=7, wall=5.62s)

- Mean±std score: **-0.11408 ± 0.0187**
- Best repeat: seed=2, score=-0.091921, `l2_regularization=1.2664e-06, learning_rate=0.085611, max_depth=9, min_samples_leaf=78`

Winning-param summary across repeats:

- `learning_rate`: median=0.085611 [q10=0.062781, q90=0.36001]
- `max_depth`: median=9 [q10=4.2, q90=13]
- `min_samples_leaf`: median=36 [q10=19.6, q90=60.6]
- `l2_regularization`: median=0.26722 [q10=8.065e-06, q90=11.708]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.11115 | `l2_regularization=0.26722, learning_rate=0.14048, max_depth=12, min_samples_leaf=35` |
| 1 | -0.11695 | `l2_regularization=1.2597e-05, learning_rate=0.061529, max_depth=5, min_samples_leaf=49` |
| 2 | -0.091921 | `l2_regularization=1.2664e-06, learning_rate=0.085611, max_depth=9, min_samples_leaf=78` |
| 3 | -0.12756 | `l2_regularization=5.0174, learning_rate=0.06521, max_depth=13, min_samples_leaf=30` |
| 4 | -0.092777 | `l2_regularization=21.744, learning_rate=0.68931, max_depth=3, min_samples_leaf=40` |
| 5 | -0.14494 | `l2_regularization=0.0058359, learning_rate=0.063616, max_depth=7, min_samples_leaf=4` |
| 6 | -0.11325 | `l2_regularization=1.2726, learning_rate=0.12289, max_depth=13, min_samples_leaf=36` |

### n_iter = 10

#### `uniform` @ n_iter=10 (n=4, wall=6.04s)

- Mean±std score: **-0.11025 ± 0.0236**
- Best repeat: seed=2, score=-0.09176, `l2_regularization=0.00011626, learning_rate=0.10951, max_depth=14, min_samples_leaf=52`

Winning-param summary across repeats:

- `learning_rate`: median=0.13689 [q10=0.11083, q90=0.17275]
- `max_depth`: median=8 [q10=2.9, q90=13.1]
- `min_samples_leaf`: median=45.5 [q10=34.1, q90=54.8]
- `l2_regularization`: median=0.00079327 [q10=8.9117e-05, q90=44.924]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.10011 | `l2_regularization=7.7484e-05, learning_rate=0.11393, max_depth=11, min_samples_leaf=56` |
| 1 | -0.14481 | `l2_regularization=64.176, learning_rate=0.17828, max_depth=5, min_samples_leaf=39` |
| 2 | -0.09176 | `l2_regularization=0.00011626, learning_rate=0.10951, max_depth=14, min_samples_leaf=52` |
| 3 | -0.10433 | `l2_regularization=0.0014703, learning_rate=0.15986, max_depth=2, min_samples_leaf=32` |

#### `lhs` @ n_iter=10 (n=4, wall=5.91s)

- Mean±std score: **-0.10225 ± 0.0139**
- Best repeat: seed=2, score=-0.085195, `l2_regularization=0.0033918, learning_rate=0.12435, max_depth=12, min_samples_leaf=65`

Winning-param summary across repeats:

- `learning_rate`: median=0.12465 [q10=0.10403, q90=0.13503]
- `max_depth`: median=6 [q10=3.3, q90=10.8]
- `min_samples_leaf`: median=52 [q10=36.2, q90=74.8]
- `l2_regularization`: median=6.6544e-06 [q10=1.0877e-07, q90=0.0023782]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.1131 | `l2_regularization=1.3161e-05, learning_rate=0.13935, max_depth=8, min_samples_leaf=35` |
| 1 | -0.11399 | `l2_regularization=1.4803e-07, learning_rate=0.12496, max_depth=4, min_samples_leaf=79` |
| 2 | -0.085195 | `l2_regularization=0.0033918, learning_rate=0.12435, max_depth=12, min_samples_leaf=65` |
| 3 | -0.09671 | `l2_regularization=9.1936e-08, learning_rate=0.095321, max_depth=3, min_samples_leaf=39` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=3, wall=13.07s)

- Mean±std score: **-0.099571 ± 0.0118**
- Best repeat: seed=2, score=-0.087491, `l2_regularization=0.014005, learning_rate=0.24058, max_depth=1, min_samples_leaf=79`

Winning-param summary across repeats:

- `learning_rate`: median=0.11393 [q10=0.099373, q90=0.21525]
- `max_depth`: median=4 [q10=1.6, q90=9.6]
- `min_samples_leaf`: median=62 [q10=57.2, q90=75.6]
- `l2_regularization`: median=7.7484e-05 [q10=1.6544e-05, q90=0.011219]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.10011 | `l2_regularization=7.7484e-05, learning_rate=0.11393, max_depth=11, min_samples_leaf=56` |
| 1 | -0.11112 | `l2_regularization=1.3084e-06, learning_rate=0.095734, max_depth=4, min_samples_leaf=62` |
| 2 | -0.087491 | `l2_regularization=0.014005, learning_rate=0.24058, max_depth=1, min_samples_leaf=79` |

#### `lhs` @ n_iter=30 (n=3, wall=13.60s)

- Mean±std score: **-0.098318 ± 0.0153**
- Best repeat: seed=2, score=-0.083823, `l2_regularization=2.9025e-07, learning_rate=0.10495, max_depth=9, min_samples_leaf=67`

Winning-param summary across repeats:

- `learning_rate`: median=0.10495 [q10=0.072039, q90=0.11771]
- `max_depth`: median=14 [q10=10, q90=14.8]
- `min_samples_leaf`: median=59 [q10=43.8, q90=65.4]
- `l2_regularization`: median=0.0015333 [q10=0.0003069, q90=0.12275]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.096819 | `l2_regularization=0.15305, learning_rate=0.1209, max_depth=15, min_samples_leaf=59` |
| 1 | -0.11431 | `l2_regularization=0.0015333, learning_rate=0.06381, max_depth=14, min_samples_leaf=40` |
| 2 | -0.083823 | `l2_regularization=2.9025e-07, learning_rate=0.10495, max_depth=9, min_samples_leaf=67` |

## `digits_logreg` — 2 tuned hparams

![neval_digits_logreg.png](figures_neval/neval_digits_logreg.png)

### Search space

```
C ~ loguniform(0.0001, 10000)
l1_ratio ~ uniform(0, 1)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -0.25023±0.0325 (n=6) | -0.25015±0.0299 (n=5) |
| 5 | -0.24753±0.0383 (n=4) | -0.23617±0.0219 (n=3) |
| 10 | -0.22856±0.0212 (n=3) | -0.22636±0.0263 (n=3) |
| 30 | -0.22469±0.0248 (n=3) | -0.2224±0.0248 (n=3) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=6, wall=5.65s)

- Mean±std score: **-0.25023 ± 0.0325**
- Best repeat: seed=1, score=-0.21667, `C=1.2433, l1_ratio=0.95046`

Winning-param summary across repeats:

- `C`: median=6.7732 [q10=1.2847, q90=134.81]
- `l1_ratio`: median=0.47932 [q10=0.27779, q90=0.83951]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.22869 | `C=12.465, l1_ratio=0.26979` |
| 1 | -0.21667 | `C=1.2433, l1_ratio=0.95046` |
| 2 | -0.2947 | `C=6.3213, l1_ratio=0.72856` |
| 3 | -0.28449 | `C=257.16, l1_ratio=0.58216` |
| 4 | -0.22725 | `C=7.2251, l1_ratio=0.37649` |
| 5 | -0.24956 | `C=1.3262, l1_ratio=0.2858` |

#### `lhs` @ n_iter=3 (n=5, wall=5.47s)

- Mean±std score: **-0.25015 ± 0.0299**
- Best repeat: seed=0, score=-0.22851, `C=14.64, l1_ratio=0.41973`

Winning-param summary across repeats:

- `C`: median=6.1508 [q10=0.26652, q90=13.91]
- `l1_ratio`: median=0.36443 [q10=0.1364, q90=0.57833]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.22851 | `C=14.64, l1_ratio=0.41973` |
| 1 | -0.23079 | `C=6.1508, l1_ratio=0.36443` |
| 2 | -0.29422 | `C=0.078094, l1_ratio=0.68406` |
| 3 | -0.26842 | `C=0.54915, l1_ratio=0.012604` |
| 4 | -0.22882 | `C=12.814, l1_ratio=0.3221` |

### n_iter = 5

#### `uniform` @ n_iter=5 (n=4, wall=6.29s)

- Mean±std score: **-0.24753 ± 0.0383**
- Best repeat: seed=0, score=-0.21599, `C=2.2336, l1_ratio=0.93507`

Winning-param summary across repeats:

- `C`: median=1.7384 [q10=0.84888, q90=5.095]
- `l1_ratio`: median=0.83182 [q10=0.33039, q90=0.94585]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.21599 | `C=2.2336, l1_ratio=0.93507` |
| 1 | -0.21667 | `C=1.2433, l1_ratio=0.95046` |
| 2 | -0.2947 | `C=6.3213, l1_ratio=0.72856` |
| 3 | -0.26276 | `C=0.67985, l1_ratio=0.15974` |

#### `lhs` @ n_iter=5 (n=3, wall=6.24s)

- Mean±std score: **-0.23617 ± 0.0219**
- Best repeat: seed=0, score=-0.21737, `C=0.1257, l1_ratio=0.45184`

Winning-param summary across repeats:

- `C`: median=0.1257 [q10=0.099098, q90=4.7454]
- `l1_ratio`: median=0.41026 [q10=0.32884, q90=0.44352]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.21737 | `C=0.1257, l1_ratio=0.45184` |
| 1 | -0.23095 | `C=5.9003, l1_ratio=0.30849` |
| 2 | -0.26018 | `C=0.092448, l1_ratio=0.41026` |

### n_iter = 10

#### `uniform` @ n_iter=10 (n=3, wall=10.13s)

- Mean±std score: **-0.22856 ± 0.0212**
- Best repeat: seed=0, score=-0.21599, `C=2.2336, l1_ratio=0.93507`

Winning-param summary across repeats:

- `C`: median=1.2433 [q10=0.35732, q90=2.0355]
- `l1_ratio`: median=0.93507 [q10=0.33682, q90=0.94739]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.21599 | `C=2.2336, l1_ratio=0.93507` |
| 1 | -0.21667 | `C=1.2433, l1_ratio=0.95046` |
| 2 | -0.25303 | `C=0.13583, l1_ratio=0.18725` |

#### `lhs` @ n_iter=10 (n=3, wall=9.76s)

- Mean±std score: **-0.22636 ± 0.0263**
- Best repeat: seed=0, score=-0.20675, `C=0.25544, l1_ratio=0.51193`

Winning-param summary across repeats:

- `C`: median=0.25544 [q10=0.18966, q90=0.87959]
- `l1_ratio`: median=0.51193 [q10=0.24742, q90=0.7864]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.20675 | `C=0.25544, l1_ratio=0.51193` |
| 1 | -0.21604 | `C=1.0356, l1_ratio=0.85501` |
| 2 | -0.25629 | `C=0.17322, l1_ratio=0.18129` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=3, wall=33.03s)

- Mean±std score: **-0.22469 ± 0.0248**
- Best repeat: seed=0, score=-0.20708, `C=0.77034, l1_ratio=0.88949`

Winning-param summary across repeats:

- `C`: median=0.76125 [q10=0.26091, q90=0.76852]
- `l1_ratio`: median=0.88949 [q10=0.3277, q90=0.96249]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.20708 | `C=0.77034, l1_ratio=0.88949` |
| 1 | -0.21395 | `C=0.76125, l1_ratio=0.98074` |
| 2 | -0.25303 | `C=0.13583, l1_ratio=0.18725` |

#### `lhs` @ n_iter=30 (n=3, wall=33.27s)

- Mean±std score: **-0.2224 ± 0.0248**
- Best repeat: seed=0, score=-0.20496, `C=0.13767, l1_ratio=0.08515`

Winning-param summary across repeats:

- `C`: median=0.10641 [q10=0.092303, q90=0.13142]
- `l1_ratio`: median=0.08515 [q10=0.020138, q90=0.17985]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.20496 | `C=0.13767, l1_ratio=0.08515` |
| 1 | -0.21151 | `C=0.10641, l1_ratio=0.0038847` |
| 2 | -0.25074 | `C=0.088776, l1_ratio=0.20352` |

## `synth_mixed_hgb` — 5 tuned hparams

![neval_synth_mixed_hgb.png](figures_neval/neval_synth_mixed_hgb.png)

### Search space

```
model__learning_rate ~ loguniform(0.001, 1)
model__max_depth ~ randint(1, 16)
model__min_samples_leaf ~ randint(1, 100)
model__l2_regularization ~ loguniform(1e-08, 50)
model__max_leaf_nodes ~ randint(8, 127)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -0.40037±0.0404 (n=10) | -0.40422±0.0671 (n=11) |
| 5 | -0.37131±0.0278 (n=7) | -0.37606±0.0187 (n=6) |
| 10 | -0.35173±0.0114 (n=4) | -0.35589±0.0169 (n=4) |
| 30 | -0.34153±0.00501 (n=3) | -0.34616±0.00366 (n=3) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=10, wall=5.36s)

- Mean±std score: **-0.40037 ± 0.0404**
- Best repeat: seed=7, score=-0.34296, `model__l2_regularization=0.53796, model__learning_rate=0.4175, model__max_depth=1, model__max_leaf_nodes=64, model__min_samples_leaf=83`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.1364 [q10=0.019422, q90=0.56911]
- `model__max_depth`: median=7 [q10=1, q90=9.5]
- `model__min_samples_leaf`: median=44 [q10=31.3, q90=92.6]
- `model__l2_regularization`: median=0.015526 [q10=2.0865e-08, q90=3.2935]
- `model__max_leaf_nodes`: median=64.5 [q10=41.7, q90=116.1]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.34872 | `model__l2_regularization=2.1171e-08, model__learning_rate=0.28026, model__max_depth=1, model__max_leaf_nodes=95, model__min_samples_leaf=86` |
| 1 | -0.39751 | `model__l2_regularization=0.44355, model__learning_rate=0.1822, model__max_depth=9, model__max_leaf_nodes=44, model__min_samples_leaf=33` |
| 2 | -0.37917 | `model__l2_regularization=0.031009, model__learning_rate=0.048618, model__max_depth=3, model__max_leaf_nodes=58, model__min_samples_leaf=44` |
| 3 | -0.42621 | `model__l2_regularization=0.13324, model__learning_rate=0.019924, model__max_depth=8, model__max_leaf_nodes=21, model__min_samples_leaf=16` |
| 4 | -0.42887 | `model__l2_regularization=6.0817e-08, model__learning_rate=0.67479, model__max_depth=9, model__max_leaf_nodes=80, model__min_samples_leaf=98` |
| 5 | -0.39921 | `model__l2_regularization=28.093, model__learning_rate=0.090596, model__max_depth=4, model__max_leaf_nodes=115, model__min_samples_leaf=44` |
| 6 | -0.37765 | `model__l2_regularization=4.2878e-05, model__learning_rate=0.041162, model__max_depth=6, model__max_leaf_nodes=126, model__min_samples_leaf=37` |
| 7 | -0.34296 | `model__l2_regularization=0.53796, model__learning_rate=0.4175, model__max_depth=1, model__max_leaf_nodes=64, model__min_samples_leaf=83` |
| 8 | -0.47404 | `model__l2_regularization=1.0898e-07, model__learning_rate=0.014902, model__max_depth=8, model__max_leaf_nodes=65, model__min_samples_leaf=38` |
| 9 | -0.42935 | `model__l2_regularization=1.8108e-08, model__learning_rate=0.55737, model__max_depth=14, model__max_leaf_nodes=60, model__min_samples_leaf=92` |

#### `lhs` @ n_iter=3 (n=11, wall=5.70s)

- Mean±std score: **-0.40422 ± 0.0671**
- Best repeat: seed=10, score=-0.34674, `model__l2_regularization=0.75743, model__learning_rate=0.11691, model__max_depth=2, model__max_leaf_nodes=47, model__min_samples_leaf=37`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.087997 [q10=0.040179, q90=0.57458]
- `model__max_depth`: median=3 [q10=2, q90=14]
- `model__min_samples_leaf`: median=71 [q10=34, q90=96]
- `model__l2_regularization`: median=0.042836 [q10=2.2702e-06, q90=1.7562]
- `model__max_leaf_nodes`: median=69 [q10=16, q90=119]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.36219 | `model__l2_regularization=0.037546, model__learning_rate=0.080811, model__max_depth=6, model__max_leaf_nodes=78, model__min_samples_leaf=95` |
| 1 | -0.51671 | `model__l2_regularization=1.7562, model__learning_rate=0.95895, model__max_depth=3, model__max_leaf_nodes=119, model__min_samples_leaf=34` |
| 2 | -0.49584 | `model__l2_regularization=0.042836, model__learning_rate=0.01663, model__max_depth=9, model__max_leaf_nodes=11, model__min_samples_leaf=96` |
| 3 | -0.3793 | `model__l2_regularization=0.0001456, model__learning_rate=0.071905, model__max_depth=2, model__max_leaf_nodes=16, model__min_samples_leaf=84` |
| 4 | -0.39051 | `model__l2_regularization=0.35278, model__learning_rate=0.040179, model__max_depth=3, model__max_leaf_nodes=111, model__min_samples_leaf=49` |
| 5 | -0.36671 | `model__l2_regularization=0.12321, model__learning_rate=0.075715, model__max_depth=14, model__max_leaf_nodes=69, model__min_samples_leaf=100` |
| 6 | -0.38042 | `model__l2_regularization=0.01477, model__learning_rate=0.29743, model__max_depth=2, model__max_leaf_nodes=16, model__min_samples_leaf=17` |
| 7 | -0.35049 | `model__l2_regularization=2.8043, model__learning_rate=0.15547, model__max_depth=2, model__max_leaf_nodes=55, model__min_samples_leaf=71` |
| 8 | -0.35113 | `model__l2_regularization=2.2702e-06, model__learning_rate=0.087997, model__max_depth=15, model__max_leaf_nodes=119, model__min_samples_leaf=91` |
| 9 | -0.5064 | `model__l2_regularization=1.646e-07, model__learning_rate=0.57458, model__max_depth=11, model__max_leaf_nodes=91, model__min_samples_leaf=67` |
| 10 | -0.34674 | `model__l2_regularization=0.75743, model__learning_rate=0.11691, model__max_depth=2, model__max_leaf_nodes=47, model__min_samples_leaf=37` |

### n_iter = 5

#### `uniform` @ n_iter=5 (n=7, wall=5.82s)

- Mean±std score: **-0.37131 ± 0.0278**
- Best repeat: seed=0, score=-0.34872, `model__l2_regularization=2.1171e-08, model__learning_rate=0.28026, model__max_depth=1, model__max_leaf_nodes=95, model__min_samples_leaf=86`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.13019 [q10=0.035827, q90=0.24818]
- `model__max_depth`: median=5 [q10=1, q90=16]
- `model__min_samples_leaf`: median=67 [q10=19, q90=86.8]
- `model__l2_regularization`: median=6.2858e-05 [q10=3.3643e-08, q90=14.057]
- `model__max_leaf_nodes`: median=67 [q10=26.4, q90=106.2]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.34872 | `model__l2_regularization=2.1171e-08, model__learning_rate=0.28026, model__max_depth=1, model__max_leaf_nodes=95, model__min_samples_leaf=86` |
| 1 | -0.35799 | `model__l2_regularization=32.519, model__learning_rate=0.17828, model__max_depth=5, model__max_leaf_nodes=123, model__min_samples_leaf=49` |
| 2 | -0.3631 | `model__l2_regularization=6.2858e-05, model__learning_rate=0.079351, model__max_depth=16, model__max_leaf_nodes=30, model__min_samples_leaf=69` |
| 3 | -0.42621 | `model__l2_regularization=0.13324, model__learning_rate=0.019924, model__max_depth=8, model__max_leaf_nodes=21, model__min_samples_leaf=16` |
| 4 | -0.35814 | `model__l2_regularization=1.967e-07, model__learning_rate=0.13019, model__max_depth=16, model__max_leaf_nodes=67, model__min_samples_leaf=67` |
| 5 | -0.3912 | `model__l2_regularization=4.1958e-08, model__learning_rate=0.046429, model__max_depth=5, model__max_leaf_nodes=89, model__min_samples_leaf=88` |
| 6 | -0.3538 | `model__l2_regularization=1.7491, model__learning_rate=0.22679, model__max_depth=1, model__max_leaf_nodes=59, model__min_samples_leaf=21` |

#### `lhs` @ n_iter=5 (n=6, wall=5.00s)

- Mean±std score: **-0.37606 ± 0.0187**
- Best repeat: seed=4, score=-0.36102, `model__l2_regularization=2.1013e-05, model__learning_rate=0.062223, model__max_depth=16, model__max_leaf_nodes=119, model__min_samples_leaf=40`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.15463 [q10=0.065018, q90=0.22876]
- `model__max_depth`: median=10.5 [q10=3.5, q90=15.5]
- `model__min_samples_leaf`: median=44.5 [q10=10.5, q90=63.5]
- `model__l2_regularization`: median=0.00010033 [q10=1.9358e-07, q90=3.6465]
- `model__max_leaf_nodes`: median=106.5 [q10=49, q90=121]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.40716 | `model__l2_regularization=7.0508, model__learning_rate=0.1307, model__max_depth=10, model__max_leaf_nodes=94, model__min_samples_leaf=1` |
| 1 | -0.39041 | `model__l2_regularization=1.1672e-07, model__learning_rate=0.24495, model__max_depth=5, model__max_leaf_nodes=123, model__min_samples_leaf=61` |
| 2 | -0.36621 | `model__l2_regularization=2.7043e-07, model__learning_rate=0.067812, model__max_depth=15, model__max_leaf_nodes=119, model__min_samples_leaf=49` |
| 3 | -0.37008 | `model__l2_regularization=0.24213, model__learning_rate=0.17856, model__max_depth=11, model__max_leaf_nodes=53, model__min_samples_leaf=66` |
| 4 | -0.36102 | `model__l2_regularization=2.1013e-05, model__learning_rate=0.062223, model__max_depth=16, model__max_leaf_nodes=119, model__min_samples_leaf=40` |
| 5 | -0.36148 | `model__l2_regularization=0.00017964, model__learning_rate=0.21257, model__max_depth=2, model__max_leaf_nodes=45, model__min_samples_leaf=20` |

### n_iter = 10

#### `uniform` @ n_iter=10 (n=4, wall=6.18s)

- Mean±std score: **-0.35173 ± 0.0114**
- Best repeat: seed=1, score=-0.33722, `model__l2_regularization=0.00028516, model__learning_rate=0.5648, model__max_depth=1, model__max_leaf_nodes=15, model__min_samples_leaf=53`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.17981 [q10=0.078177, q90=0.47944]
- `model__max_depth`: median=3 [q10=1, q90=12.7]
- `model__min_samples_leaf`: median=72 [q10=57.8, q90=82.7]
- `model__l2_regularization`: median=0.00017401 [q10=1.8872e-05, q90=0.070775]
- `model__max_leaf_nodes`: median=32 [q10=19.5, q90=76.7]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.34872 | `model__l2_regularization=2.1171e-08, model__learning_rate=0.28026, model__max_depth=1, model__max_leaf_nodes=95, model__min_samples_leaf=86` |
| 1 | -0.33722 | `model__l2_regularization=0.00028516, model__learning_rate=0.5648, model__max_depth=1, model__max_leaf_nodes=15, model__min_samples_leaf=53` |
| 2 | -0.3631 | `model__l2_regularization=6.2858e-05, model__learning_rate=0.079351, model__max_depth=16, model__max_leaf_nodes=30, model__min_samples_leaf=69` |
| 3 | -0.35788 | `model__l2_regularization=0.10098, model__learning_rate=0.077673, model__max_depth=5, model__max_leaf_nodes=34, model__min_samples_leaf=75` |

#### `lhs` @ n_iter=10 (n=4, wall=6.49s)

- Mean±std score: **-0.35589 ± 0.0169**
- Best repeat: seed=0, score=-0.3354, `model__l2_regularization=0.00046885, model__learning_rate=0.35279, model__max_depth=1, model__max_leaf_nodes=10, model__min_samples_leaf=32`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.1494 [q10=0.069353, q90=0.31435]
- `model__max_depth`: median=9 [q10=2.8, q90=11.7]
- `model__min_samples_leaf`: median=66.5 [q10=38.9, q90=78.7]
- `model__l2_regularization`: median=0.01953 [q10=0.0034514, q90=0.079245]
- `model__max_leaf_nodes`: median=56 [q10=20.2, q90=73.6]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.3354 | `model__l2_regularization=0.00046885, model__learning_rate=0.35279, model__max_depth=1, model__max_leaf_nodes=10, model__min_samples_leaf=32` |
| 1 | -0.35383 | `model__l2_regularization=0.10093, model__learning_rate=0.067295, model__max_depth=12, model__max_leaf_nodes=68, model__min_samples_leaf=78` |
| 2 | -0.37648 | `model__l2_regularization=0.028649, model__learning_rate=0.22465, model__max_depth=11, model__max_leaf_nodes=44, model__min_samples_leaf=79` |
| 3 | -0.35785 | `model__l2_regularization=0.010411, model__learning_rate=0.074156, model__max_depth=7, model__max_leaf_nodes=76, model__min_samples_leaf=55` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=3, wall=14.14s)

- Mean±std score: **-0.34153 ± 0.00501**
- Best repeat: seed=1, score=-0.33601, `model__l2_regularization=1.1389e-08, model__learning_rate=0.11858, model__max_depth=3, model__max_leaf_nodes=39, model__min_samples_leaf=40`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.53688 [q10=0.20224, q90=0.72213]
- `model__max_depth`: median=1 [q10=1, q90=2.6]
- `model__min_samples_leaf`: median=40 [q10=24.8, q90=74.4]
- `model__l2_regularization`: median=0.00010686 [q10=2.138e-05, q90=0.0023326]
- `model__max_leaf_nodes`: median=100 [q10=51.2, q90=105.6]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.34578 | `model__l2_regularization=0.00010686, model__learning_rate=0.53688, model__max_depth=1, model__max_leaf_nodes=107, model__min_samples_leaf=83` |
| 1 | -0.33601 | `model__l2_regularization=1.1389e-08, model__learning_rate=0.11858, model__max_depth=3, model__max_leaf_nodes=39, model__min_samples_leaf=40` |
| 2 | -0.34281 | `model__l2_regularization=0.002889, model__learning_rate=0.76844, model__max_depth=1, model__max_leaf_nodes=100, model__min_samples_leaf=21` |

#### `lhs` @ n_iter=30 (n=3, wall=14.71s)

- Mean±std score: **-0.34616 ± 0.00366**
- Best repeat: seed=0, score=-0.3426, `model__l2_regularization=2.0005, model__learning_rate=0.36253, model__max_depth=1, model__max_leaf_nodes=26, model__min_samples_leaf=70`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.36253 [q10=0.15263, q90=0.72165]
- `model__max_depth`: median=1 [q10=1, q90=4.2]
- `model__min_samples_leaf`: median=70 [q10=53.2, q90=86]
- `model__l2_regularization`: median=0.0082026 [q10=0.0016407, q90=1.602]
- `model__max_leaf_nodes`: median=70 [q10=34.8, q90=86]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.3426 | `model__l2_regularization=2.0005, model__learning_rate=0.36253, model__max_depth=1, model__max_leaf_nodes=26, model__min_samples_leaf=70` |
| 1 | -0.34595 | `model__l2_regularization=2.5489e-07, model__learning_rate=0.10016, model__max_depth=5, model__max_leaf_nodes=90, model__min_samples_leaf=49` |
| 2 | -0.34991 | `model__l2_regularization=0.0082026, model__learning_rate=0.81142, model__max_depth=1, model__max_leaf_nodes=70, model__min_samples_leaf=90` |

## `synth_mixed_logreg` — 2 tuned hparams

![neval_synth_mixed_logreg.png](figures_neval/neval_synth_mixed_logreg.png)

### Search space

```
model__C ~ loguniform(0.0001, 10000)
model__l1_ratio ~ uniform(0, 1)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -0.31973±0.0417 (n=47) | -0.31216±0.00439 (n=47) |
| 5 | -0.31217±0.00417 (n=29) | -0.3117±0.00413 (n=28) |
| 10 | -0.31225±0.00427 (n=14) | -0.31173±0.00399 (n=14) |
| 30 | -0.31288±0.00277 (n=5) | -0.31303±0.00283 (n=5) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=47, wall=5.00s)

- Mean±std score: **-0.31973 ± 0.0417**
- Best repeat: seed=46, score=-0.30359, `model__C=2966.4, model__l1_ratio=0.084947`

Winning-param summary across repeats:

- `model__C`: median=6.6863 [q10=0.37597, q90=861.31]
- `model__l1_ratio`: median=0.43888 [q10=0.079624, q90=0.94828]

| seed | score | winning params |
|-----:|------:|----------------|
| 2 | -0.30879 | `model__C=6.3213, model__l1_ratio=0.72856` |
| 7 | -0.31083 | `model__C=10.018, model__l1_ratio=0.89721` |
| 8 | -0.30616 | `model__C=910.27, model__l1_ratio=0.39108` |
| 10 | -0.3084 | `model__C=1.266, model__l1_ratio=0.13592` |
| 11 | -0.31191 | `model__C=6.4861, model__l1_ratio=0.028689` |
| 13 | -0.31152 | `model__C=828.67, model__l1_ratio=0.8553` |
| 14 | -0.30985 | `model__C=13.506, model__l1_ratio=0.54836` |
| 15 | -0.30578 | `model__C=3.7392, model__l1_ratio=0.14625` |
| 17 | -0.31125 | `model__C=2.8971, model__l1_ratio=0.36808` |
| 21 | -0.31045 | `model__C=11.11, model__l1_ratio=0.98081` |
| 22 | -0.30786 | `model__C=0.47282, model__l1_ratio=0.98768` |
| 23 | -0.30977 | `model__C=16.856, model__l1_ratio=0.85346` |
| 24 | -0.30747 | `model__C=3.2636, model__l1_ratio=0.56969` |
| 26 | -0.30871 | `model__C=319.83, model__l1_ratio=0.54834` |
| 29 | -0.3058 | `model__C=1.4252, model__l1_ratio=0.2652` |
| 30 | -0.31052 | `model__C=182.64, model__l1_ratio=0.86758` |
| 33 | -0.30686 | `model__C=5.1316, model__l1_ratio=0.35912` |
| 36 | -0.31071 | `model__C=29.11, model__l1_ratio=0.51108` |
| 37 | -0.30944 | `model__C=42.976, model__l1_ratio=0.66167` |
| 38 | -0.31083 | `model__C=0.76929, model__l1_ratio=0.24974` |
| 39 | -0.30855 | `model__C=1.032, model__l1_ratio=0.53934` |
| 40 | -0.30566 | `model__C=3430.6, model__l1_ratio=0.059652` |
| 43 | -0.30921 | `model__C=4.979, model__l1_ratio=0.22471` |
| 45 | -0.31255 | `model__C=1.2073, model__l1_ratio=0.77934` |
| 46 | -0.30359 | `model__C=2966.4, model__l1_ratio=0.084947` |

_Showing 25/47 repeats (highest scores). Full list in JSON._

#### `lhs` @ n_iter=3 (n=47, wall=5.02s)

- Mean±std score: **-0.31216 ± 0.00439**
- Best repeat: seed=46, score=-0.30325, `model__C=2.7426, model__l1_ratio=0.71835`

Winning-param summary across repeats:

- `model__C`: median=36.687 [q10=1.4173, q90=2562.3]
- `model__l1_ratio`: median=0.50059 [q10=0.1928, q90=0.92696]

| seed | score | winning params |
|-----:|------:|----------------|
| 2 | -0.30916 | `model__C=4072.2, model__l1_ratio=0.35043` |
| 7 | -0.30992 | `model__C=1.4342, model__l1_ratio=0.28948` |
| 8 | -0.30578 | `model__C=5.9694, model__l1_ratio=0.891` |
| 10 | -0.30997 | `model__C=316.97, model__l1_ratio=0.68928` |
| 12 | -0.30694 | `model__C=116.16, model__l1_ratio=0.37096` |
| 13 | -0.31162 | `model__C=7406.5, model__l1_ratio=0.50059` |
| 14 | -0.31017 | `model__C=1435.8, model__l1_ratio=0.3767` |
| 15 | -0.3055 | `model__C=1.3434, model__l1_ratio=0.86864` |
| 19 | -0.30991 | `model__C=0.41675, model__l1_ratio=0.8677` |
| 21 | -0.31061 | `model__C=1037.6, model__l1_ratio=0.77529` |
| 22 | -0.30973 | `model__C=6.2485, model__l1_ratio=0.15354` |
| 23 | -0.30986 | `model__C=37.432, model__l1_ratio=0.44632` |
| 24 | -0.30803 | `model__C=344.19, model__l1_ratio=0.47641` |
| 25 | -0.31181 | `model__C=508.88, model__l1_ratio=0.75105` |
| 26 | -0.30873 | `model__C=7563.2, model__l1_ratio=0.18587` |
| 29 | -0.30603 | `model__C=2625, model__l1_ratio=0.50112` |
| 30 | -0.30995 | `model__C=5.2524, model__l1_ratio=0.2085` |
| 33 | -0.3072 | `model__C=1839.8, model__l1_ratio=0.23965` |
| 36 | -0.30966 | `model__C=1.5979, model__l1_ratio=0.32193` |
| 37 | -0.30938 | `model__C=21.447, model__l1_ratio=0.71486` |
| 38 | -0.31102 | `model__C=36.687, model__l1_ratio=0.46284` |
| 39 | -0.30897 | `model__C=6.6415, model__l1_ratio=0.19742` |
| 40 | -0.30569 | `model__C=367.27, model__l1_ratio=0.69024` |
| 43 | -0.30978 | `model__C=17.727, model__l1_ratio=0.93512` |
| 46 | -0.30325 | `model__C=2.7426, model__l1_ratio=0.71835` |

_Showing 25/47 repeats (highest scores). Full list in JSON._

### n_iter = 5

#### `uniform` @ n_iter=5 (n=29, wall=5.18s)

- Mean±std score: **-0.31217 ± 0.00417**
- Best repeat: seed=15, score=-0.30579, `model__C=3.7392, model__l1_ratio=0.14625`

Winning-param summary across repeats:

- `model__C`: median=6.3213 [q10=1.1306, q90=65.928]
- `model__l1_ratio`: median=0.44869 [q10=0.13175, q90=0.93815]

| seed | score | winning params |
|-----:|------:|----------------|
| 1 | -0.31387 | `model__C=1.2433, model__l1_ratio=0.95046` |
| 2 | -0.30879 | `model__C=6.3213, model__l1_ratio=0.72856` |
| 3 | -0.31441 | `model__C=0.67985, model__l1_ratio=0.15974` |
| 5 | -0.31272 | `model__C=1.3262, model__l1_ratio=0.2858` |
| 6 | -0.31495 | `model__C=2.0198, model__l1_ratio=0.34327` |
| 7 | -0.31083 | `model__C=10.018, model__l1_ratio=0.89721` |
| 8 | -0.30614 | `model__C=910.27, model__l1_ratio=0.39108` |
| 10 | -0.3084 | `model__C=1.266, model__l1_ratio=0.13592` |
| 11 | -0.31193 | `model__C=6.4861, model__l1_ratio=0.028689` |
| 12 | -0.30681 | `model__C=23.098, model__l1_ratio=0.11508` |
| 13 | -0.31107 | `model__C=8.1345, model__l1_ratio=0.0026308` |
| 14 | -0.30938 | `model__C=0.5462, model__l1_ratio=0.57246` |
| 15 | -0.30579 | `model__C=3.7392, model__l1_ratio=0.14625` |
| 16 | -0.31635 | `model__C=3.4303, model__l1_ratio=0.43074` |
| 17 | -0.31124 | `model__C=2.8971, model__l1_ratio=0.36808` |
| 18 | -0.31361 | `model__C=14.275, model__l1_ratio=0.57687` |
| 19 | -0.31344 | `model__C=176.9, model__l1_ratio=0.5387` |
| 20 | -0.31349 | `model__C=35.69, model__l1_ratio=0.44869` |
| 21 | -0.31045 | `model__C=11.11, model__l1_ratio=0.98081` |
| 22 | -0.30786 | `model__C=0.47282, model__l1_ratio=0.98768` |
| 23 | -0.30981 | `model__C=16.856, model__l1_ratio=0.85346` |
| 24 | -0.30744 | `model__C=3.2636, model__l1_ratio=0.56969` |
| 25 | -0.31184 | `model__C=8450.8, model__l1_ratio=0.78693` |
| 26 | -0.30821 | `model__C=2.7205, model__l1_ratio=0.17581` |
| 28 | -0.31636 | `model__C=1.8234, model__l1_ratio=0.78513` |

_Showing 25/29 repeats (highest scores). Full list in JSON._

#### `lhs` @ n_iter=5 (n=28, wall=5.03s)

- Mean±std score: **-0.3117 ± 0.00413**
- Best repeat: seed=8, score=-0.30561, `model__C=3.5662, model__l1_ratio=0.30011`

Winning-param summary across repeats:

- `model__C`: median=3.354 [q10=0.70259, q90=39.362]
- `model__l1_ratio`: median=0.45202 [q10=0.20917, q90=0.95803]

| seed | score | winning params |
|-----:|------:|----------------|
| 1 | -0.31608 | `model__C=5.9003, model__l1_ratio=0.30849` |
| 2 | -0.30911 | `model__C=95.764, model__l1_ratio=0.7303` |
| 3 | -0.31434 | `model__C=0.69793, model__l1_ratio=0.20756` |
| 5 | -0.31269 | `model__C=1.2777, model__l1_ratio=0.39965` |
| 6 | -0.31563 | `model__C=5.301, model__l1_ratio=0.077899` |
| 7 | -0.30996 | `model__C=1.0019, model__l1_ratio=0.68261` |
| 8 | -0.30561 | `model__C=3.5662, model__l1_ratio=0.30011` |
| 10 | -0.30955 | `model__C=13.433, model__l1_ratio=0.27554` |
| 11 | -0.31064 | `model__C=0.90841, model__l1_ratio=0.35414` |
| 12 | -0.30624 | `model__C=0.62258, model__l1_ratio=0.88221` |
| 13 | -0.31088 | `model__C=3.5454, model__l1_ratio=0.83795` |
| 14 | -0.31012 | `model__C=35.863, model__l1_ratio=0.65347` |
| 15 | -0.30619 | `model__C=47.525, model__l1_ratio=0.72118` |
| 16 | -0.31541 | `model__C=1.3578, model__l1_ratio=0.40416` |
| 17 | -0.31021 | `model__C=0.98163, model__l1_ratio=0.47237` |
| 18 | -0.3109 | `model__C=0.64092, model__l1_ratio=0.98985` |
| 19 | -0.31222 | `model__C=6.1996, model__l1_ratio=0.12519` |
| 20 | -0.31235 | `model__C=1.5384, model__l1_ratio=0.74709` |
| 21 | -0.30965 | `model__C=3.5045, model__l1_ratio=0.24239` |
| 22 | -0.30765 | `model__C=0.70459, model__l1_ratio=0.96984` |
| 23 | -0.30838 | `model__C=2.3354, model__l1_ratio=0.99121` |
| 24 | -0.30743 | `model__C=1.2506, model__l1_ratio=0.43167` |
| 25 | -0.31114 | `model__C=6.3606, model__l1_ratio=0.95298` |
| 26 | -0.30856 | `model__C=11.706, model__l1_ratio=0.22902` |
| 27 | -0.31852 | `model__C=0.87771, model__l1_ratio=0.42367` |

_Showing 25/28 repeats (highest scores). Full list in JSON._

### n_iter = 10

#### `uniform` @ n_iter=10 (n=14, wall=5.00s)

- Mean±std score: **-0.31225 ± 0.00427**
- Best repeat: seed=8, score=-0.3058, `model__C=6.8262, model__l1_ratio=0.64686`

Winning-param summary across repeats:

- `model__C`: median=1.2961 [q10=0.59324, q90=5.4849]
- `model__l1_ratio`: median=0.31454 [q10=0.08072, q90=0.93281]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.31841 | `model__C=2.2336, model__l1_ratio=0.93507` |
| 1 | -0.31387 | `model__C=1.2433, model__l1_ratio=0.95046` |
| 2 | -0.30842 | `model__C=3.1487, model__l1_ratio=0.15006` |
| 3 | -0.31441 | `model__C=0.67985, model__l1_ratio=0.15974` |
| 4 | -0.31669 | `model__C=0.27795, model__l1_ratio=0.78895` |
| 5 | -0.31271 | `model__C=1.3262, model__l1_ratio=0.2858` |
| 6 | -0.31492 | `model__C=2.0198, model__l1_ratio=0.34327` |
| 7 | -0.31 | `model__C=1.0874, model__l1_ratio=0.5535` |
| 8 | -0.3058 | `model__C=6.8262, model__l1_ratio=0.64686` |
| 9 | -0.31925 | `model__C=0.7578, model__l1_ratio=0.065154` |
| 10 | -0.3084 | `model__C=1.266, model__l1_ratio=0.13592` |
| 11 | -0.3119 | `model__C=6.4861, model__l1_ratio=0.028689` |
| 12 | -0.30619 | `model__C=0.55613, model__l1_ratio=0.92752` |
| 13 | -0.31053 | `model__C=1.3867, model__l1_ratio=0.11704` |

#### `lhs` @ n_iter=10 (n=14, wall=5.01s)

- Mean±std score: **-0.31173 ± 0.00399**
- Best repeat: seed=8, score=-0.30514, `model__C=1.1032, model__l1_ratio=0.74769`

Winning-param summary across repeats:

- `model__C`: median=1.2834 [q10=0.77019, q90=2.9589]
- `model__l1_ratio`: median=0.76145 [q10=0.22128, q90=0.93663]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.31758 | `model__C=1.3957, model__l1_ratio=0.27134` |
| 1 | -0.31374 | `model__C=1.0356, model__l1_ratio=0.85501` |
| 2 | -0.30836 | `model__C=1.811, model__l1_ratio=0.77819` |
| 3 | -0.31302 | `model__C=0.81822, model__l1_ratio=0.94578` |
| 4 | -0.31301 | `model__C=0.60891, model__l1_ratio=0.976` |
| 5 | -0.31262 | `model__C=2.8393, model__l1_ratio=0.19983` |
| 6 | -0.31517 | `model__C=3.0102, model__l1_ratio=0.70589` |
| 7 | -0.30945 | `model__C=1.4075, model__l1_ratio=0.91529` |
| 8 | -0.30514 | `model__C=1.1032, model__l1_ratio=0.74769` |
| 9 | -0.31873 | `model__C=1.6877, model__l1_ratio=0.019704` |
| 10 | -0.3083 | `model__C=1.1711, model__l1_ratio=0.7374` |
| 11 | -0.31212 | `model__C=5.3564, model__l1_ratio=0.80033` |
| 12 | -0.30618 | `model__C=1.0296, model__l1_ratio=0.77521` |
| 13 | -0.31073 | `model__C=0.74961, model__l1_ratio=0.71897` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=5, wall=5.37s)

- Mean±std score: **-0.31288 ± 0.00277**
- Best repeat: seed=2, score=-0.30836, `model__C=1.3571, model__l1_ratio=0.59344`

Winning-param summary across repeats:

- `model__C`: median=0.77034 [q10=0.6582, q90=1.216]
- `model__l1_ratio`: median=0.88949 [q10=0.66538, q90=0.97188]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.31566 | `model__C=0.77034, model__l1_ratio=0.88949` |
| 1 | -0.31277 | `model__C=0.76125, model__l1_ratio=0.98074` |
| 2 | -0.30836 | `model__C=1.3571, model__l1_ratio=0.59344` |
| 3 | -0.31323 | `model__C=0.58949, model__l1_ratio=0.77328` |
| 4 | -0.3144 | `model__C=1.0042, model__l1_ratio=0.95858` |

#### `lhs` @ n_iter=30 (n=5, wall=5.34s)

- Mean±std score: **-0.31303 ± 0.00283**
- Best repeat: seed=2, score=-0.30832, `model__C=1.69, model__l1_ratio=0.61114`

Winning-param summary across repeats:

- `model__C`: median=0.85269 [q10=0.5824, q90=1.4113]
- `model__l1_ratio`: median=0.93961 [q10=0.67506, q90=0.95857]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.31575 | `model__C=0.85269, model__l1_ratio=0.9676` |
| 1 | -0.31344 | `model__C=0.99338, model__l1_ratio=0.93961` |
| 2 | -0.30832 | `model__C=1.69, model__l1_ratio=0.61114` |
| 3 | -0.31313 | `model__C=0.49561, model__l1_ratio=0.94501` |
| 4 | -0.31451 | `model__C=0.7126, model__l1_ratio=0.77092` |

## `credit_g_hgb` — 5 tuned hparams

![neval_credit_g_hgb.png](figures_neval/neval_credit_g_hgb.png)

### Search space

```
model__learning_rate ~ loguniform(0.001, 1)
model__max_depth ~ randint(1, 16)
model__min_samples_leaf ~ randint(1, 100)
model__l2_regularization ~ loguniform(1e-08, 50)
model__max_leaf_nodes ~ randint(8, 127)
```

| n_iter | Uniform mean±std (n) | LHS mean±std (n) |
|------:|----------------------:|-----------------:|
| 3 | -0.51877±0.0142 (n=7) | -0.52361±0.0175 (n=9) |
| 5 | -0.51373±0.00816 (n=6) | -0.52248±0.0182 (n=5) |
| 10 | -0.50643±0.00738 (n=3) | -0.5084±0.0117 (n=3) |
| 30 | -0.50447±0.00852 (n=3) | -0.49967±0.00836 (n=3) |

### Winning hyperparameter combos

### n_iter = 3

#### `uniform` @ n_iter=3 (n=7, wall=5.13s)

- Mean±std score: **-0.51877 ± 0.0142**
- Best repeat: seed=0, score=-0.50268, `model__l2_regularization=1.4464e-08, model__learning_rate=0.081449, model__max_depth=5, model__max_leaf_nodes=105, model__min_samples_leaf=5`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.041162 [q10=0.017344, q90=0.1157]
- `model__max_depth`: median=6 [q10=4, q90=14.2]
- `model__min_samples_leaf`: median=16 [q10=5.6, q90=39.8]
- `model__l2_regularization`: median=0.13324 [q10=2.7924e-06, q90=20.767]
- `model__max_leaf_nodes`: median=86 [q10=35.4, q90=119.4]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.50268 | `model__l2_regularization=1.4464e-08, model__learning_rate=0.081449, model__max_depth=5, model__max_leaf_nodes=105, model__min_samples_leaf=5` |
| 1 | -0.52958 | `model__l2_regularization=15.883, model__learning_rate=0.034313, model__max_depth=16, model__max_leaf_nodes=45, model__min_samples_leaf=15` |
| 2 | -0.50529 | `model__l2_regularization=4.6443e-06, model__learning_rate=0.15335, model__max_depth=4, model__max_leaf_nodes=86, model__min_samples_leaf=6` |
| 3 | -0.51869 | `model__l2_regularization=0.13324, model__learning_rate=0.019924, model__max_depth=8, model__max_leaf_nodes=21, model__min_samples_leaf=16` |
| 4 | -0.54376 | `model__l2_regularization=2.8442, model__learning_rate=0.013473, model__max_depth=13, model__max_leaf_nodes=73, model__min_samples_leaf=18` |
| 5 | -0.51729 | `model__l2_regularization=28.093, model__learning_rate=0.090596, model__max_depth=4, model__max_leaf_nodes=115, model__min_samples_leaf=44` |
| 6 | -0.51411 | `model__l2_regularization=4.2878e-05, model__learning_rate=0.041162, model__max_depth=6, model__max_leaf_nodes=126, model__min_samples_leaf=37` |

#### `lhs` @ n_iter=3 (n=9, wall=5.64s)

- Mean±std score: **-0.52361 ± 0.0175**
- Best repeat: seed=0, score=-0.5082, `model__l2_regularization=0.037546, model__learning_rate=0.080811, model__max_depth=6, model__max_leaf_nodes=78, model__min_samples_leaf=95`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.075715 [q10=0.015783, q90=0.18386]
- `model__max_depth`: median=6 [q10=2, q90=14.2]
- `model__min_samples_leaf`: median=91 [q10=42.6, q90=96.8]
- `model__l2_regularization`: median=0.037546 [q10=1.8266e-06, q90=0.84308]
- `model__max_leaf_nodes`: median=55 [q10=10.4, q90=112.6]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.5082 | `model__l2_regularization=0.037546, model__learning_rate=0.080811, model__max_depth=6, model__max_leaf_nodes=78, model__min_samples_leaf=95` |
| 1 | -0.5568 | `model__l2_regularization=5.1975e-08, model__learning_rate=0.012396, model__max_depth=13, model__max_leaf_nodes=8, model__min_samples_leaf=91` |
| 2 | -0.54883 | `model__l2_regularization=0.042836, model__learning_rate=0.01663, model__max_depth=9, model__max_leaf_nodes=11, model__min_samples_leaf=96` |
| 3 | -0.51964 | `model__l2_regularization=0.0001456, model__learning_rate=0.071905, model__max_depth=2, model__max_leaf_nodes=16, model__min_samples_leaf=84` |
| 4 | -0.52406 | `model__l2_regularization=0.35278, model__learning_rate=0.040179, model__max_depth=3, model__max_leaf_nodes=111, model__min_samples_leaf=49` |
| 5 | -0.51392 | `model__l2_regularization=0.12321, model__learning_rate=0.075715, model__max_depth=14, model__max_leaf_nodes=69, model__min_samples_leaf=100` |
| 6 | -0.50889 | `model__l2_regularization=0.01477, model__learning_rate=0.29743, model__max_depth=2, model__max_leaf_nodes=16, model__min_samples_leaf=17` |
| 7 | -0.51117 | `model__l2_regularization=2.8043, model__learning_rate=0.15547, model__max_depth=2, model__max_leaf_nodes=55, model__min_samples_leaf=71` |
| 8 | -0.52101 | `model__l2_regularization=2.2702e-06, model__learning_rate=0.087997, model__max_depth=15, model__max_leaf_nodes=119, model__min_samples_leaf=91` |

### n_iter = 5

#### `uniform` @ n_iter=5 (n=6, wall=6.04s)

- Mean±std score: **-0.51373 ± 0.00816**
- Best repeat: seed=0, score=-0.50268, `model__l2_regularization=1.4464e-08, model__learning_rate=0.081449, model__max_depth=5, model__max_leaf_nodes=105, model__min_samples_leaf=5`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.1104 [q10=0.050687, q90=0.16581]
- `model__max_depth`: median=5 [q10=4, q90=12]
- `model__min_samples_leaf`: median=30 [q10=5.5, q90=58]
- `model__l2_regularization`: median=0.066623 [q10=1.0558e-07, q90=30.306]
- `model__max_leaf_nodes`: median=95.5 [q10=44, q90=119]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.50268 | `model__l2_regularization=1.4464e-08, model__learning_rate=0.081449, model__max_depth=5, model__max_leaf_nodes=105, model__min_samples_leaf=5` |
| 1 | -0.5238 | `model__l2_regularization=32.519, model__learning_rate=0.17828, model__max_depth=5, model__max_leaf_nodes=123, model__min_samples_leaf=49` |
| 2 | -0.50529 | `model__l2_regularization=4.6443e-06, model__learning_rate=0.15335, model__max_depth=4, model__max_leaf_nodes=86, model__min_samples_leaf=6` |
| 3 | -0.51869 | `model__l2_regularization=0.13324, model__learning_rate=0.019924, model__max_depth=8, model__max_leaf_nodes=21, model__min_samples_leaf=16` |
| 4 | -0.51466 | `model__l2_regularization=1.967e-07, model__learning_rate=0.13019, model__max_depth=16, model__max_leaf_nodes=67, model__min_samples_leaf=67` |
| 5 | -0.51729 | `model__l2_regularization=28.093, model__learning_rate=0.090596, model__max_depth=4, model__max_leaf_nodes=115, model__min_samples_leaf=44` |

#### `lhs` @ n_iter=5 (n=5, wall=5.71s)

- Mean±std score: **-0.52248 ± 0.0182**
- Best repeat: seed=0, score=-0.50473, `model__l2_regularization=7.0508, model__learning_rate=0.1307, model__max_depth=10, model__max_leaf_nodes=94, model__min_samples_leaf=1`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.062223 [q10=0.018358, q90=0.10554]
- `model__max_depth`: median=15 [q10=8.8, q90=16]
- `model__min_samples_leaf`: median=49 [q10=16.6, q90=74.6]
- `model__l2_regularization`: median=0.031501 [q10=8.5676e-06, q90=4.8025]
- `model__max_leaf_nodes`: median=94 [q10=70.6, q90=119]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.50473 | `model__l2_regularization=7.0508, model__learning_rate=0.1307, model__max_depth=10, model__max_leaf_nodes=94, model__min_samples_leaf=1` |
| 1 | -0.5493 | `model__l2_regularization=0.031501, model__learning_rate=0.016001, model__max_depth=8, model__max_leaf_nodes=69, model__min_samples_leaf=91` |
| 2 | -0.50961 | `model__l2_regularization=2.7043e-07, model__learning_rate=0.067812, model__max_depth=15, model__max_leaf_nodes=119, model__min_samples_leaf=49` |
| 3 | -0.532 | `model__l2_regularization=1.4301, model__learning_rate=0.021892, model__max_depth=16, model__max_leaf_nodes=73, model__min_samples_leaf=50` |
| 4 | -0.51678 | `model__l2_regularization=2.1013e-05, model__learning_rate=0.062223, model__max_depth=16, model__max_leaf_nodes=119, model__min_samples_leaf=40` |

### n_iter = 10

#### `uniform` @ n_iter=10 (n=3, wall=5.57s)

- Mean±std score: **-0.50643 ± 0.00738**
- Best repeat: seed=0, score=-0.49969, `model__l2_regularization=1.894e-05, model__learning_rate=0.051831, model__max_depth=6, model__max_leaf_nodes=54, model__min_samples_leaf=60`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.15335 [q10=0.072135, q90=0.48251]
- `model__max_depth`: median=4 [q10=1.6, q90=5.6]
- `model__min_samples_leaf`: median=53 [q10=15.4, q90=58.6]
- `model__l2_regularization`: median=1.894e-05 [q10=7.5035e-06, q90=0.00023191]
- `model__max_leaf_nodes`: median=54 [q10=22.8, q90=79.6]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.49969 | `model__l2_regularization=1.894e-05, model__learning_rate=0.051831, model__max_depth=6, model__max_leaf_nodes=54, model__min_samples_leaf=60` |
| 1 | -0.51431 | `model__l2_regularization=0.00028516, model__learning_rate=0.5648, model__max_depth=1, model__max_leaf_nodes=15, model__min_samples_leaf=53` |
| 2 | -0.50529 | `model__l2_regularization=4.6443e-06, model__learning_rate=0.15335, model__max_depth=4, model__max_leaf_nodes=86, model__min_samples_leaf=6` |

#### `lhs` @ n_iter=10 (n=3, wall=6.40s)

- Mean±std score: **-0.5084 ± 0.0117**
- Best repeat: seed=2, score=-0.50088, `model__l2_regularization=4.7134, model__learning_rate=0.056069, model__max_depth=7, model__max_leaf_nodes=33, model__min_samples_leaf=20`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.067295 [q10=0.058314, q90=0.29569]
- `model__max_depth`: median=7 [q10=2.2, q90=11]
- `model__min_samples_leaf`: median=32 [q10=22.4, q90=68.8]
- `model__l2_regularization`: median=0.10093 [q10=0.020561, q90=3.7909]
- `model__max_leaf_nodes`: median=33 [q10=14.6, q90=61]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.5024 | `model__l2_regularization=0.00046885, model__learning_rate=0.35279, model__max_depth=1, model__max_leaf_nodes=10, model__min_samples_leaf=32` |
| 1 | -0.52192 | `model__l2_regularization=0.10093, model__learning_rate=0.067295, model__max_depth=12, model__max_leaf_nodes=68, model__min_samples_leaf=78` |
| 2 | -0.50088 | `model__l2_regularization=4.7134, model__learning_rate=0.056069, model__max_depth=7, model__max_leaf_nodes=33, model__min_samples_leaf=20` |

### n_iter = 30

#### `uniform` @ n_iter=30 (n=3, wall=17.44s)

- Mean±std score: **-0.50447 ± 0.00852**
- Best repeat: seed=2, score=-0.49942, `model__l2_regularization=0.0020977, model__learning_rate=0.047658, model__max_depth=12, model__max_leaf_nodes=42, model__min_samples_leaf=20`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.051831 [q10=0.048492, q90=0.4622]
- `model__max_depth`: median=6 [q10=2, q90=10.8]
- `model__min_samples_leaf`: median=53 [q10=26.6, q90=58.6]
- `model__l2_regularization`: median=0.00028516 [q10=7.2184e-05, q90=0.0017352]
- `model__max_leaf_nodes`: median=42 [q10=20.4, q90=51.6]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.49969 | `model__l2_regularization=1.894e-05, model__learning_rate=0.051831, model__max_depth=6, model__max_leaf_nodes=54, model__min_samples_leaf=60` |
| 1 | -0.51431 | `model__l2_regularization=0.00028516, model__learning_rate=0.5648, model__max_depth=1, model__max_leaf_nodes=15, model__min_samples_leaf=53` |
| 2 | -0.49942 | `model__l2_regularization=0.0020977, model__learning_rate=0.047658, model__max_depth=12, model__max_leaf_nodes=42, model__min_samples_leaf=20` |

#### `lhs` @ n_iter=30 (n=3, wall=18.61s)

- Mean±std score: **-0.49967 ± 0.00836**
- Best repeat: seed=2, score=-0.4909, `model__l2_regularization=8.4195e-06, model__learning_rate=0.11083, model__max_depth=3, model__max_leaf_nodes=115, model__min_samples_leaf=6`

Winning-param summary across repeats:

- `model__learning_rate`: median=0.071688 [q10=0.052532, q90=0.10301]
- `model__max_depth`: median=11 [q10=4.6, q90=11.8]
- `model__min_samples_leaf`: median=9 [q10=6.6, q90=18.6]
- `model__l2_regularization`: median=0.44872 [q10=0.08975, q90=10.417]
- `model__max_leaf_nodes`: median=14 [q10=12.4, q90=94.8]

| seed | score | winning params |
|-----:|------:|----------------|
| 0 | -0.50056 | `model__l2_regularization=12.909, model__learning_rate=0.071688, model__max_depth=11, model__max_leaf_nodes=14, model__min_samples_leaf=9` |
| 1 | -0.50756 | `model__l2_regularization=0.44872, model__learning_rate=0.047743, model__max_depth=12, model__max_leaf_nodes=12, model__min_samples_leaf=21` |
| 2 | -0.4909 | `model__l2_regularization=8.4195e-06, model__learning_rate=0.11083, model__max_depth=3, model__max_leaf_nodes=115, model__min_samples_leaf=6` |

Full per-repeat winner JSON: `experiments/lhs_vs_uniform/results/winners/`.

This pull request includes code written with the assistance of AI.
The code has **not yet been reviewed** by a human.
