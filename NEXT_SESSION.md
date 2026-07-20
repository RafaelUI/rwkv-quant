# Continuation prompt — rwkv-quant

Квантование RWKV-7 под Apple Silicon (M4 base, 16GB, БЕЗВЕНТИЛЯТОРНЫЙ Air).
Цель: максимальная производительность decode при минимальной потере качества.
Формат `.rwkvq`, бэкенд Metal (MLX), референс-модель
`~/Develop/WKV-kvant/rwkv7-g1h-1.5b-ctx10240.pth` (1.5B, world naming, ~2.8GB).
Полный лог сессий до 19.07 (вечер, КЕРНЕЛЬ-3) включительно — в git-истории.

## Законы методологии (нарушались, приводило к ложным выводам)

1. Все замеры скорости — ТОЛЬКО A/B-чередованием в одном процессе.
   Безвентиляторный дрейф под нагрузкой до 1.8x на «том же» замере.
2. Тяжёлые ppl-прогоны 1.5B — по одному конфигу на процесс
   (`tests/diagnose_one.py <case>`); не тащить лишние модели в процесс.
3. ppl сравнивать только внутри текущего корпуса
   (eval_corpus_world.pt[:8] из ~/Develop/test.txt).
4. В циклах с деквантом — eval каждую итерацию.
5. Комбинации конфигов непредсказуемы: изолированный выигрыш не
   переносится в композит без замера. Отрицательные результаты не удалять.
6. Metal System Trace: точные числа — через `xcrun xctrace export` +
   union интервалов, не «Frame» в GUI.
7. eval_corpus_world.pt строки по 128 токенов: `data[:, :1024]` МОЛЧА
   даёт T=128. «Слишком хорошие» числа сверять с руфлайном
   (полоса 120 GB/s = пол decode ~7 мс/ток на 843MB).
8. mx-массивы иммутабельны: снапшот state = хранение ссылки.
9. НОВОЕ (19.07-k3): в mx.fast.metal_kernel `threadgroup_position_in_grid`
   — это ИНДЕКС threadgroup (grid задаётся в потоках, но позиция — в
   группах, не делить на размер TG). И: проверка корректности нового
   кернеля сравнением массивов может дать ЛОЖНЫЙ бит-экзакт — MLX
   рециклит выходные буферы, непокрытые гридом строки наследуют
   значения предыдущего (референсного!) вызова. Сначала доказать
   покрытие grid'а, потом верить BIT=.

## КЕРНЕЛЬ-3 — СДЕЛАН (19.07 вечер, в проде, бит-в-бит)

quant_linear_gw.py, K3=True (боевой путь; старый кернель цел — fallback).
Три составляющих, дисковый формат НЕТРОНУТ (репак при загрузке):
1. Раскладка MLX qmv: NSG simdgroups x RS строк на TG (было 1x32, R=8).
   Диспетч по форме: _k3_cfg (N=1) / _k3_cfg_nb (N-батч), свипы в
   tests/bench_kernel3_protoD.py / protoE.py.
2. Интерлив: qblk = codes16Б[+qh4Б[+qh2 4Б]] контигуозно на блок;
   qsqm=uchar2/блок; ddm=half2/суперблок. 4-5 транзакций вместо 7.
   Память 1x: старые имена (codes/qs/…) — ленивые view (__getattr__).
3. Мульт-трюк битплоскостей: (nib*0x00204081)&0x01010101 — ~3x меньше
   ALU на плоскость. Разблокировал int6 (был 1.00-1.04, стал 1.12-1.25).
Порядок математики на lane сохранён => БИТ-В-БИТ со старым кернелем на
всех реальных формах (ad-hoc гейт в логе сессии), все гейты зелёные
(test_gw_kernel[_int6], test_gw_nb_parity, test_fuse_parity, smoke,
parity, v2_format, streaming, combined). ppl НЕ перемерялся — не нужен.

Микро (GB/s, старый→к3): tmix5 41→50, tmix6 47→59, cmixK4 65→76,
cmixK6 71→80, cmixV4 69→73, cmixV6 69→78, head5 96→101, head6 88→103.
N-батч NN=4: tmix x1.19-1.20, cmixK6 x1.34, head x1.40.

## Текущее состояние: пресеты (decode/verify перемерены 19.07-k3, A/B)

| | размер | ppl | vs bf16 | decode (compiled) | prefill T=1024 |
|---|---|---|---|---|---|
| bf16 | 2953MB | 11.430 | — | — | — |
| COMPRESSION (gw sb6+AW, 4/5 бит) | 970.7MB | 11.7125 | +2.47% | **14.8** (было 16.2); FUSE 14.15; spec+ngram 14.02 | 545 t/s (не задет) |
| REDUCTION v2 (sb6 int6, AW кроме proj) | 1255.9MB | 11.4438 | +0.12% | **17.7** (было 20.3) | 437 |
| MollySophia MLX INT6 (G1!) | 1272.0MB | 11.5507* | +1.06%* | 15.10 | 591 |

*через границу чекпоинтов. РАЗРЫВ С MOLLY ЗАКРЫТ: чемпион теперь
быстрее (14.8 vs 15.1) при родном качестве — как и предсказывал вывод
19.07 («скорость добывать кернелем при родном формате»).
Verify (raw forward_stateful, A/B): чемпион T=4 37.1→32.4, T=8 65.2→58.5;
reduction T=4 42.3→35.6, T=8 73.3→64.4.
Спекулятивка чемпион+ngram k=4: 14.02 мс/ток (87% приёмка, цикл. текст).
Отчёт Бо Пэну: `~/Develop/int6_comparison_report.md` (числа decode там
СТАРЫЕ — если ещё не отправлен, обновить).

## Закрытые темы (не переоткрывать без новых данных)

- ANE/CoreML; GPU idle (занятость 98.99%); qs/qm-упаковка в кернеле;
  пайплайн decode; 4x-сжатие в нибл-формате; MXFP4; v3 simdgroup head;
  nb2 (подъём загрузок); GEMM r/k/v-фьюз префилла; гибрид mlx-affine
  ради скорости (качество не окупает) — всё как раньше, см. git.
- НОВОЕ: xbsum-в-кернеле (K3_XSUM, за флагом=False): decode x1.005,
  verify x0.94, логиты дрейфуют до 0.02 абс. Тупик.
- НОВОЕ: ПОЛУБЛОК (K3_HALF, за флагом=False, _get_kernel_k3h в коде):
  микро tmix +8-19%, но E2E НОЛЬ (чемпион x1.003, reduction x1.006) —
  малые кернели не на критическом пути. Перм плоскостей не нужен
  (dyn-сдвиг не хуже, protoF3). Не включать без новых данных.
- НОВОЕ: packs2 (2 смежных блока/поток) — хуже базы (x0.5-0.85).
- НОВОЕ: свипы (NSG,RS) для cmix вплоть до (1,1)/(8,4) — в тепловом
  шуме (±5-9%), стену 75-80 не двигают. «x1.26 s1r4» был артефактом
  троттлинга (закон 1!).

## Профиль decode чемпиона (19.07 вечер-2, аблации compiled, ~14.9мс)

cmix 5.54 | LoRA 2.33 | head 1.02 | WKV 0.88 | tmix+обвязка остальное.
Вывод дня: ускорение МАЛЫХ кернелей (tmix) в e2e НЕ видно (прячутся за
конвейером запусков), ускорение КРУПНЫХ (cmix/head) переносится ~1:1.
cmix-стена: наши 75-80 GB/s vs MLX 94.7 НА ТОЙ ЖЕ ФОРМЕ (перемерено
bench_mlx_vs_gw_ab с к3; tmix 70 vs 87). Это разрыв В КЕРНЕЛЕ:
floor вызова кастомного кернеля = нативного (5.5-10 мкс, tiny-probe),
30-мкс теории оверхеда мертвы. Свипы (NSG,RS) от (1,1) до (8,4) и
packs2 — плоско либо хуже (packs2 x0.5-0.85, отрицательный).

## Вечер-3 (19.07): карта пределов decode ЗАМКНУТА

- КЕРНЕЛЬ-4 ЗАКРЫТ руфлайн-пробой (bench_kernel4_roofline.py):
  mem-only вариант (та же память, ноль декода) = full в пределах 2-7%
  на всех cmix-формах => декодный ALU бесплатен, premult-qdot и любые
  ALU-трюки бессмысленны. premul-вариант написан и равен full.
- «Разрыв с MLX на cmix» БЫЛ МИРАЖОМ закона 1: при строгом
  почередном A/B на остывшей машине MLX qmm на тех же формах =
  72-81 GB/s = наш пол (утренние 94.7 -- тепловой артефакт
  квази-последовательного bench_mlx_vs_gw_ab).
- LORA_Q8 (int8-лоры в фьюзе, mx.quantized_matmul, код за флагом в
  quant_model.py): decode x1.004, дрейф логитов 0.77 абс. =>
  ОТРИЦАТЕЛЬНЫЙ. Лора-блок латентно-bound (мелкие опы), не трафик.
- Вывод: decode ~14.5 (FUSE) упирается в ЧИСЛО ОПОВ на токен
  (~250-350 запусков), не в полосу/ALU. Большие кернели (cmix/head)
  на полу паттерна памяти, мелкие прячутся за конвейером.

## Открытые вопросы / следующие шаги (приоритет по цене/выгоде)

1. ОП-КАУНТ: фьюзия лора-блока в 1-2 кастомных кернеля на слой
   (wav-down+act+up одним запуском, one-TG или two-stage с барьером;
   g-пара так же). Сейчас ~6-8 опов/слой на лорах. Оценка -0.5-1 мс.
   Того же поля: fuse groupnorm+bonus+gate вокруг WKV.
2. Решение А.: FUSE=True по умолчанию? Фьюз стабильно -0.8 мс
   (14.48 vs 15.31, гейты зелёные, ppl 11.7129 vs 11.7125 -- разница
   путей, задокументирована в test_fuse_parity).
3. Верификационная полка: T>=12 ~7.3-7.9 мс/ток; перемерить кривую
   (bench_verify_cost_compiled). Верифай-раунды тоже оп-каунт-bound
   (см. полублок/NB: кернельные выигрыши там были, полка от обвязки).
4. REDUCTION докстринги/CLI перепроверить перед релизом;
   eval_v2_real/eval_v2_kernel оформить из прототипов.
5. nb-режим для quant_linear_v2 (g_lora/small int8): ~1-2 мс на
   verify-раунд.
6. Драфт-спекулятивка: после кернеля-4; n-gram уже можно в прод
   (1.08-1.25x, проигрыша нет).
7. AW-автоправило; canonical INT6 на G1 bf16; гетерогенный GPU+CPU —
   понижены.

## QLoRA поверх .rwkvq (REDUCTION) в rwkv-metal — 19-20.07 (два дня, свежая ветка)

Отдельная задача от кернель-3: rwkv-metal (`~/Develop/rwkv-metal/`) — НЕ
только импорт WKV7-кернелей, там уже был полный LoRA/QLoRA-стек
(`rwkv_metal/lora/`: `add_lora`, `finetune`, grad-checkpoint по блокам,
дифференцируемый WKV7 backward, `nn.value_and_grad` + `freeze()`,
провалидированный рецепт). Но QLoRA там был на стоковом `mlx.nn.quantize`
(generic groupwise-affine), не на нашем sb6/asym. Задача: завести QLoRA
именно на REDUCTION-пресете (специально откалиброван под это, см.
presets.py: "деградация около нуля, для QAT/QLoRA-базы") -- по мотивам
Unsloth-подхода (кастомный квант + LoRA вместе, а не generic bnb).

### Что сделано (всё бит-в-бит сверено с rwkv_quant.formats.reader)

1. **MLX-порт sb6/asym dequant** (`rwkv_metal/lora/rwkvq_linear.py`,
   `rwkvq_kernel.py`) -- 0 расхождений на реальных reduction_v2.rwkvq/
   champion_v2.rwkvq тензорах (bits 4/5/6, sb6 и asym режимы, до 134M
   элементов). Формат: см. rwkv_quant/formats/schema.py (split-ниббл
   pack_nib_block, битплоскости pack_bitplane для 5/6 бит, суперблок
   6-бит qs/qm против fp16 d/dm). Финальный combine ОБЯЗАТЕЛЬНО в
   float32 (не half!) -- `GwQuantLinear._dequant_w()` (rwkv-quant, для
   GEMM-префилла) держит математику в half и даёт ~18% расхождений на
   1 бит бф16-мантиссы -- НЕ бит-в-бит, для QLoRA-базы неприемлемо.
2. **Мост без torch в рантайме**: rwkv-metal принципиально torch-free
   (`model/convert.py`), а .rwkvq читается только через torch
   (`rwkv_quant/formats/reader.py`). Решение -- `rwkv_quant/formats/
   export_mlx.py` (venv rwkv-quant, есть torch): once конвертирует
   .rwkvq -> `*.rwkvq_mlx.safetensors` + `.json`-манифест, используя
   K3-интерлив-буферы (qblk/qsqm/ddm) из `backends/metal/
   quant_linear_gw.py::GwQuantLinear` -- те же буферы, что и в
   проверенном бит-в-бит инференс-пути. rwkv-metal грузит сайдкар через
   голый `mx.load` + json, без torch.
3. **Интеграция**: `rwkv_metal/lora/lora.py::LoRALinear` получил
   `base_module=` (обходит `nn.QuantizedLinear.from_linear`, принимает
   готовый frozen-модуль); `add_rwkvq.py::add_lora_rwkvq`/
   `load_lora_rwkvq_model` -- обвязка. Соответствие имён: `.rwkvq` уже в
   официальной x070-схеме (та же, что жрёт `model/convert.py::convert()`
   -- REDUCTION/COMPRESSION калибровались на официальном чекпоинте),
   маппинг тривиальный.
4. **Ленивая загрузка** (`model/convert.py::load_pretrained_partial`,
   `_load_pth_lazy`, `_LazyTensor`): не читать+decompress байты .pth для
   тензоров (proj/cmix/head), которые тут же заменяются сжатыми --
   pre_materialize_hook подменяет их СРАЗУ после конструирования
   скелета модели, ДО материализации остального bf16 (порядок вызовов
   критичен -- первая версия, где хук шёл ПОСЛЕ, давала пик ХУЖЕ, не
   лучше, чем без ленивой загрузки вообще -- см. ниже).
5. **Fused Metal-кернель** (`rwkvq_kernel.py::dequant_dense`) -- один
   `mx.fast.metal_kernel` launch вместо ~8 отдельных MLX-операций,
   embarrassingly parallel по (row, block), без simd-редукции (в
   отличие от K3 GEMV -- тут просто dequant, не dot-product). 3.8-7.1x
   быстрее композитного MLX-порта, бит-в-бит идентичен.
6. **Родная MLX-упаковка** (`rwkvq_native.py`) -- ключевая находка:
   MLX ставится С ИСХОДНИКАМИ Metal-кернелей (`<venv>/lib/.../mlx/
   include/mlx/backend/metal/kernels/quantized.h`, 2596 строк,
   `affine_qmm_t` -- настоящий тайловый GEMM с fused-декантом,
   `QuantizedBlockLoader` внутри). in ПРЯМОЕ переиспользование `mx.
   quantize()` НЕ подходит -- пересчитывает scale/bias по min/max блока,
   ~89% значений расходятся (round-trip реквантование поверх уже
   квантованного, `dev_check_requantize_roundtrip.py`). Вместо этого:
   реверс-инжинирили точную битовую раскладку `wq` эмпирически (one-hot
   тесты, `dev_reverse_mlx_pack.py`) -- LSB-first битовый поток на
   группу из 32, поле позиции p с глобального бита p*6, переходит через
   границы uint32-слов без выравнивания; формула `w=code*scale+bias`
   идентична нашей. Вручную упаковали НАШИ точные sb6-коды+scale+bias в
   их контейнер -> `mx.dequantize`/`mx.quantized_matmul` воспроизводят
   исходные значения бит-в-бит (0 расхождений). Работает ТОЛЬКО для
   bits=6 (REDUCTION) -- quantized.h ветвит паковку для степеней двойки
   иначе (`get_bytes_per_pack`), для bits=4/5 (COMPRESSION) не
   проверялось.
7. **Гибрид** (`rwkvq_hybrid.py`) -- коды в родной MLX-раскладке (wq) +
   компактные sb6 scale/bias (qsqm/ddm), разворачиваемые на лету перед
   `quantized_matmul`. Гипотеза была "память как у fused-кернеля,
   скорость как у native" -- НЕ подтвердилась чисто (см. таблицу).

### Замеры (реальная 1.5B модель rwkv7-g1h-1.5b-ctx10240.pth, M4 base,
    B=1 T=128, grad_checkpoint=True, LoRA rank=16 на 4 tmix-проекциях x
    24 слоя, REDUCTION-пресет)

| путь                         | шаг (steady) | mx_peak (train) | ru_maxrss (реальный) |
|-------------------------------|:---:|:---:|:---:|
| stock QLoRA (nn.quantize 6bit)| 0.80-0.89с | 3.02GB | ~6.9GB (eager .pth) |
| rwkvq, наивный MLX-порт       | 3.4-3.6с   | 4.39GB | 6.6-7.0GB |
| rwkvq, fused-кернель          | 1.2-1.3с   | 3.65GB | 3.64GB (лениво) |
| rwkvq, native quantized_matmul| 0.7-0.8с   | 4.59GB | 4.36GB (лениво) |
| rwkvq, гибрид                 | 0.8-0.9с   | 4.19GB | 4.62GB (лениво) |

Выводы:
- **native == stock по скорости** (буквально тот же кернель MLX), при
  памяти всё равно заметно лучше "наивной" загрузки за счёт ленивого
  загрузчика (п.4) -- но memory overhead родного формата (fp32 scale+bias
  ПОЛНЫМ per-group-32, ~0.25Б/значение) больше, чем у sb6-суперблочной
  compact-схемы (~0.012Б/значение) -- родной путь тяжелее fused-кернеля
  на ~700MB-1GB.
- **fused-кернель** -- лучший баланс память/риск (свой код, не зависит
  от внутренностей MLX), но на 1.5x медленнее пика скорости.
- **Гибрид не оправдался**: разворачивание compact scale/bias (~12-15
  мелких MLX-операций на маленьких [OUT,NB]-массивах) упирается в ту же
  болезнь launch-overhead, что и самый первый наивный порт -- на
  маленьких tmix-проекциях ДАЖЕ МЕДЛЕННЕЕ native (2.83мс vs 1.2мс на
  2048x2048), реальный RSS в лучшем случае наравне с native, не ниже.
  Чтобы гибрид реально выиграл, разворачивание scale/bias тоже нужно
  зашить в один маленький fused-кернель -- не делали (ROI под вопросом,
  делянка небольшая: scale/bias -- считанные % от веса тензора).
- Порядок вызовов в ленивой загрузке КРИТИЧЕН: hook (замена proj/cmix/
  head на сжатые) должен идти СРАЗУ после конструирования скелета
  модели, ДО материализации остального bf16 -- иначе случайная
  bf16-инициализация (полного размера, из `RWKV7X070.__init__`) и
  реальные данные временно сосуществуют в памяти, пик получается ХУЖЕ,
  чем вообще без ленивой загрузки (замерено: 5.74GB против 1.50GB после
  перестановки).
- `mx.get_peak_memory()` НЕ отражает реальный RSS процесса -- проверено
  на разнице ru_maxrss (реальный high-water-mark) vs mx-трекнутый пик:
  загрузка ОДНОГО 3GB bf16 .pth через torch-free zip-загрузчик
  (`load_pth`) сама по себе даёт ~6.9GB RSS (2.3x размера файла,
  промежуточные zip/pickle-буферы) -- это ОБЩАЯ проблема для стокового
  и rwkvq-путей, не специфична для нашей работы.

### Незакрытое / на будущее

- `emb.weight` НЕ квантуется (ни в одном из путей) -- `nn.Embedding`
  делает gather по индексам, не x@W^T, `RwkvqLinear`/`Native`/`Hybrid`
  рассчитаны на Linear-семантику. Нужен отдельный indexed-gather путь
  (dequant только нужных строк, не всей таблицы 65536x2048).
- native/hybrid проверены ТОЛЬКО для bits=6 (REDUCTION). COMPRESSION
  (proj=4/5bit, cmix=4bit) через них не проверялся -- нужен отдельный
  реверс битовой раскладки MLX для не-степень-двойки/степень-двойки
  битности (`get_bytes_per_pack` в quantized.h ветвит логику).
- Гибридный fused-кернель для scale/bias (закрыл бы разрыв гибрида) --
  не делали, под вопросом ROI.
- ppl/качество на реальном датасете с QLoRA-адаптерами после нескольких
  сотен/тысяч шагов -- НЕ мерили (только 4 синтетических шага на
  случайных токенах, чисто для проверки forward/backward/memory/speed,
  loss убывает но это не показатель качества).
- COMPRESSION-пресет (970MB) для QLoRA-базы не пробовали вообще --
  сессия целиком на REDUCTION (он и задуман под QLoRA).

### Где лежит

- `~/Develop/rwkv-metal/rwkv_metal/lora/`: `rwkvq_linear.py` (fused-
  кернель путь + `_dequant_w_slow` референс), `rwkvq_kernel.py` (Metal
  dequant-кернель), `rwkvq_native.py` (родная MLX-упаковка), `rwkvq_
  hybrid.py` (гибрид), `add_rwkvq.py` (`add_lora_rwkvq`/
  `load_lora_rwkvq_model`, флаг `native=True|False|"hybrid"`).
- `~/Develop/rwkv-metal/rwkv_metal/model/convert.py`: `_load_pth_lazy`,
  `_LazyTensor`, `load_pretrained_partial` (ленивая загрузка).
- `~/Develop/rwkv-quant/rwkv_quant/formats/export_mlx.py`: экспортёр
  сайдкара (запускать в venv rwkv-quant после пересборки .rwkvq --
  `/tmp` не переживает ребут).
- Тесты/бенчи (оба репо, `tests/dev*.py` -- НЕ гейты, песочница):
  `mlx_dequant_proto/sweep/hotpath/reuse_gw/precise_fast.py` (rwkv-
  quant) -- корректность+скорость наивного порта по стадиям;
  `dev_bench_stock_qlora.py`, `dev_bench_rwkvq_qlora/mem/rss.py`,
  `dev_bench_rwkvq_lazyload_rss.py`, `dev_check_rwkvq_linear.py`,
  `dev_check_rwkvq_fused_kernel.py`, `dev_reverse_mlx_pack.py`,
  `dev_pack_native_mlx.py`, `dev_bench_native_qmm.py`, `dev_check_
  hybrid.py`, `dev_bench_hybrid_full.py` (rwkv-metal).
- Артефакты в /tmp (не переживают ребут, пересобрать): `reduction_v2.
  rwkvq_mlx.safetensors`+`.json` (сайдкар), `ref_bits_check.
  safetensors` (эталонные bf16-биты для кросс-венв сверки).


## Где лежат материалы

- `~/Develop/WKV-kvant/` — чекпоинты (1.5B G1H, драфт g1d-0.1b, ru60m),
  корпуса (eval_corpus_world.pt [24x128!], .txt), world_tokenizer.py.
- `~/Develop/rwkv7-1.5B-g1g-mlx-6bit/` — чекпоинт Molly + словарь.
- `~/Develop/rwkv-metal/` — WKV7 Metal-кернели + LoRA/QLoRA-стек;
  19-20.07 добавлен QLoRA поверх .rwkvq (rwkv_metal/lora/rwkvq_*.py,
  см. раздел выше) -- теперь ТРОГАЕМ, не только импорт.
- `/tmp` (пересобрать после ребута!): champion_v2.rwkvq (970.7MB),
  reduction_v2.rwkvq (1255.9MB), canonical_int4/int6.rwkvq,
  act_stats_1p5b.pt.
- tests/: venv, diagnose_one.py, гейты (test_gw_nb_parity,
  test_gw_kernel[_int6], test_fuse_parity, test_v2_*), бенчи
  (bench_verify_cost*, bench_gw_nb_*, bench_mlx_vs_gw_*, 4way),
  КЕРНЕЛЬ-3: bench_kernel3_proto.py (фаза B, реструктуризация),
  protoC (интерлив), protoD (мульт-трюк; свип N=1), protoE (N-батч),
  protoF/F2/F3 (полублок: perm/dyn/static, e2e-ноль), bench_k3_e2e_ab.py
  (e2e A/B old-vs-new, аргумент — .rwkvq), bench_k3h_e2e_ab.py
  (A/B флага K3_HALF), bench_k3_cmix_sweep.py (packs2, отрицательный),
  profile_decode_v2_compiled.py (аблации: cmix/LoRA/head/WKV),
  спекулятивка (spec_decode_poc/champ/draft/draft2.py),
  eval_mlx_affine_real.py (prefill-печать с ловушкой закона 7 —
  игнорировать), eval_molly_real.py, eval_canonical_int4/int6.py.
