# Synthetic BN Benchmark V1 Devam Planı

Bu dosya yeni bir chatte kaldığımız yerden devam etmek için oluşturuldu.

## Mevcut Durum

- Repo: `/home/oguzhan/causal-portfolio-selector`
- Synthetic dataset root: `data/synthetic_bn/v1`
- Synthetic run root: `artifacts/synthetic_runs/v1`
- Synthetic dataset üretimi tamamlandı:
  - `data/synthetic_bn/v1/manifest.json`
  - `2000` dataset CSV
  - `2000` ground-truth JSON
- Algorithm run yarıda kesildi.
- Son kontrol edilen durum:
  - Toplam beklenen kayıt: `2000 × 9 = 18000`
  - Mevcut kayıt: `8809`
  - Başarılı run: `8569`
  - Timeout: `240`
  - Tamamlanan dataset: `976`
  - Kısmi dataset: `4`
  - Henüz başlanmamış dataset: `1020`
- Timeoutların tamamı şu ana kadar `MMHC` tarafında göründü.

Önemli: `run-synthetic-algorithms` resume-safe çalışır. Var olan JSON kayıtlarını `--resume` ile atlar. Devam ederken `--overwrite` kullanma.

## 1. Kaldığın Yerden Algorithm Run'a Devam Et

4 ayrı tmux session aç. Her session aynı output klasörünü kullanacak, sadece `--shard-index` değişecek.

Shard 0:

```bash
tmux new -s synthetic_algos_0
cd /home/oguzhan/causal-portfolio-selector
uv run --extra benchmark causal-portfolio --config configs/default.yaml run-synthetic-algorithms \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_runs/v1 \
  --timeout-seconds 300 \
  --shard-index 0 \
  --shard-count 4 \
  --resume
```

Shard 1:

```bash
tmux new -s synthetic_algos_1
cd /home/oguzhan/causal-portfolio-selector
uv run --extra benchmark causal-portfolio --config configs/default.yaml run-synthetic-algorithms \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_runs/v1 \
  --timeout-seconds 300 \
  --shard-index 1 \
  --shard-count 4 \
  --resume
```

Shard 2:

```bash
tmux new -s synthetic_algos_2
cd /home/oguzhan/causal-portfolio-selector
uv run --extra benchmark causal-portfolio --config configs/default.yaml run-synthetic-algorithms \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_runs/v1 \
  --timeout-seconds 300 \
  --shard-index 2 \
  --shard-count 4 \
  --resume
```

Shard 3:

```bash
tmux new -s synthetic_algos_3
cd /home/oguzhan/causal-portfolio-selector
uv run --extra benchmark causal-portfolio --config configs/default.yaml run-synthetic-algorithms \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_runs/v1 \
  --timeout-seconds 300 \
  --shard-index 3 \
  --shard-count 4 \
  --resume
```

Tmux'tan çıkmak için:

```bash
Ctrl-b d
```

Tekrar bağlanmak için:

```bash
tmux attach -t synthetic_algos_0
```

## 2. Run Progress Kontrolü

Kayıt sayısı:

```bash
cd /home/oguzhan/causal-portfolio-selector
find artifacts/synthetic_runs/v1/records -type f | wc -l
```

Beklenen final sayı:

```text
18000
```

Detaylı status özeti:

```bash
cd /home/oguzhan/causal-portfolio-selector
uv run python - <<'PY'
from pathlib import Path
import json
import pandas as pd

records = list(Path("artifacts/synthetic_runs/v1/records").glob("*.json"))
rows = []
for path in records:
    record = json.loads(path.read_text())
    rows.append({
        "dataset_name": record.get("dataset_name"),
        "algorithm_name": record.get("algorithm_name"),
        "status": record.get("status"),
        "runtime_seconds": record.get("runtime_seconds"),
    })

df = pd.DataFrame(rows)
print("record_count", len(records))
if not df.empty:
    print("\nstatus counts")
    print(df["status"].value_counts(dropna=False).to_string())
    print("\nalgorithm x status")
    print(pd.crosstab(df["algorithm_name"], df["status"]).to_string())
    completed = df.groupby("dataset_name")["algorithm_name"].nunique().eq(9).sum()
    partial = df.groupby("dataset_name")["algorithm_name"].nunique().between(1, 8).sum()
    untouched = 2000 - df["dataset_name"].nunique()
    print("\ncompleted datasets", completed)
    print("partial datasets", partial)
    print("untouched datasets", untouched)
PY
```

Aktif process kontrolü:

```bash
pgrep -af 'run-synthetic-algorithms|causal-portfolio' || true
```

## 3. Algorithm Run Bitince Training Table Oluştur

Tüm shard'lar bittiğinde bu komutu çalıştır:

```bash
cd /home/oguzhan/causal-portfolio-selector
causal-portfolio --config configs/default.yaml build-synthetic-training-tables \
  --synthetic-root data/synthetic_bn/v1 \
  --runs artifacts/synthetic_runs/v1 \
  --output artifacts/synthetic_tables/v1
```

Bu komut `summary.csv` dosyasına ihtiyaç duymaz; doğrudan `records/` altındaki JSON kayıtlarını okur.

Beklenen çıktılar:

- `artifacts/synthetic_tables/v1/features.csv`
- `artifacts/synthetic_tables/v1/targets.csv`
- `artifacts/synthetic_tables/v1/run_evaluations.csv`
- `artifacts/synthetic_tables/v1/splits.csv`

## 4. Synthetic Encoder Eğit

GPU kullanılacak:

```bash
cd /home/oguzhan/causal-portfolio-selector
uv run --with torch causal-portfolio --config configs/default.yaml train-fingerprint-from-synthetic \
  --synthetic-root data/synthetic_bn/v1 \
  --output artifacts/synthetic_models/v1/biaffine_encoder.pt \
  --device cuda \
  --epochs 100
```

Beklenen çıktılar:

- `artifacts/synthetic_models/v1/biaffine_encoder.pt`
- `artifacts/synthetic_models/v1/biaffine_encoder.manifest.json`

## 5. Synthetic Selector Eğit

```bash
cd /home/oguzhan/causal-portfolio-selector
causal-portfolio --config configs/default.yaml train-synthetic-selector \
  --tables artifacts/synthetic_tables/v1 \
  --encoder artifacts/synthetic_models/v1/biaffine_encoder.pt \
  --output artifacts/synthetic_models/v1/selector.joblib
```

Bu aşamada üç feature set denenir:

- `handcrafted_all`
- `learned_only`
- `handcrafted_plus_learned`

Model seçimi yalnızca synthetic validation sonuçlarıyla yapılır. Exact bnlearn datasetleri model seçimi için kullanılmaz.

Beklenen çıktılar:

- `artifacts/synthetic_models/v1/selector.joblib`
- `artifacts/synthetic_models/v1/selector_manifest.json`
- `artifacts/synthetic_models/v1/selector_validation_metrics.csv`
- `artifacts/synthetic_models/v1/selector_validation_summary.csv`
- `artifacts/synthetic_models/v1/selector_test_metrics.csv`
- `artifacts/synthetic_models/v1/selector_test_summary.csv`

## 6. Final Exact Evaluation Çalıştır

Bu aşama synthetic-trained modeli 15 exact bnlearn dataset üzerinde test eder.

```bash
cd /home/oguzhan/causal-portfolio-selector
causal-portfolio --config configs/default.yaml evaluate-synthetic-selector-on-exact \
  --model artifacts/synthetic_models/v1/selector.joblib \
  --encoder artifacts/synthetic_models/v1/biaffine_encoder.pt \
  --output reports/synthetic_v1
```

Beklenen çıktılar:

- `reports/synthetic_v1/exact_metrics.csv`
- `reports/synthetic_v1/exact_summary.csv`
- `reports/synthetic_v1/exact_predictions.csv`
- `reports/synthetic_v1/exact_report.md`

## 7. Kabul Kriterleri

- `data/synthetic_bn/v1/manifest.json` içinde `2000` dataset olmalı.
- `artifacts/synthetic_runs/v1/records` altında finalde `18000` JSON record beklenir.
- Timeout kayıtları sorun değildir; training table oluşturma aşaması timeoutları unavailable kabul eder.
- Exact bnlearn datasetler training/validation için kullanılmamalı.
- Final exact raporda synthetic selector ile eski Phase 3 fixed policy karşılaştırılmalı.

## 8. Sorun Çıkarsa

- Eğer shard kesilirse aynı shard komutunu tekrar çalıştır. `--resume` var olan kayıtları atlar.
- Eğer bir process takılı kalırsa önce kontrol et:

```bash
pgrep -af 'run-synthetic-algorithms|causal-portfolio'
```

- Gerekirse sadece ilgili takılı process'i öldür:

```bash
kill <PID>
```

- Sonra aynı shard komutunu tekrar başlat.
