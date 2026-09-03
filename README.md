# MetaDrive RL (Stable-Baselines3)

MetaDrive 環境を Stable-Baselines3 の PPO で学習・評価するためのプロジェクトです。学習・評価などの Python コマンドは、この README があるプロジェクトルートで実行します。 `configs/official.toml` は、MetaDrive公式の Stable-Baselines3 サンプル相当の例です。

## 環境構築

`requirements.txt` は `../metadrive` を editable install します。このリポジトリと `metadrive/` を同じ親ディレクトリの直下に配置してください。

```text
<workspace>/
├── metadrive/
└── <this-project>/
```

`metadrive/` をまだ取得していない場合は、このリポジトリを含む親ディレクトリで MetaDrive公式リポジトリを clone します。

```bash
git clone https://github.com/metadriverse/metadrive.git metadrive
```

次に、この README があるプロジェクトルートへ移動して、仮想環境と依存関係を作成します。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m metadrive.pull_asset
```

最後のコマンドは MetaDrive の実行に必要な asset を取得します。この処理にはネットワーク接続が必要です。Python 3.12 で動作を確認しています。

## 学習方法

`configs/official.toml` は MetaDrive公式サンプルと同じ学習タスクと主要 PPO パラメータを使う例です。`map = "C"`、3×3 の離散 Action、`horizon = 500`、scenario seed 5 の1 scenario、4並列環境、`MlpPolicy`、`n_steps = 4096`、`total_timesteps = 300000` を指定し、実行 device は `cpu` に固定しています。

```bash
.venv/bin/python train.py --config configs/official.toml
```

学習が完了すると、モデルは `models/official_baseline.zip`、実行メタデータは `outputs/official/training/official_baseline/training_metadata.json` に保存されます。Monitor ログと TensorBoard ログはそれぞれ `logs/monitor/`、`logs/tensorboard/` に保存されます。

## 評価方法

公式サンプル相当の設定で学習したモデルを評価します。

```bash
.venv/bin/python evaluate.py --config configs/official.toml
```

この設定では `models/official_baseline.zip` を読み込み、`outputs/official/evaluation/official_baseline/` に `evaluation.json`、`evaluation_steps.jsonl`、可視化成果物を保存します。可視化を出力しない場合は次を実行します。

```bash
.venv/bin/python evaluate.py --config configs/official.toml --no-record-gif
```

別のモデルを指定する場合は、`--model models/<model-name>.zip` を追加します。

## 入力寄与・入力依存度解析

固定済み PPO `MlpPolicy` の vector observation に対して、baseline 置換による摂動依存度と Integrated Gradients (IG) を解析できます。学習、`evaluate.py`、MetaDrive 本体は変更しません。実験の目的・数式・設定変更箇所は [入力寄与解析ガイド](docs/input_attribution.md)、出力の絞り方と解釈例は [結果の読み方](docs/input_attribution_results.md)、259 次元 schema の成立条件は [観測スキーマの説明](observation_schemas/README.md)、262 次元環境への変更は [移植手順](docs/input_attribution_porting.md) を確認してください。

```bash
.venv/bin/python analyze_input_attribution.py run \
  --config configs/official.toml \
  --model models/official_baseline.zip \
  --schema observation_schemas/metadrive_default_259.toml \
  --analysis-config attribution_configs/official_left_curve.toml \
  --output-prefix official_left_curve
```

成功した解析結果は `outputs/official/attribution/official_left_curve/` に、rollout、数値表、plot、`report.md` として保存されます。`analyze` は保存済み rollout の再解析だけを行うため MetaDrive を再起動しません。custom 262 次元版では `custom_262_template.toml` の unresolved placeholder を、確認済みの index・意味・group に置き換えた後に `validate-schema` を実行してください。

## configファイル

設定は TOML で記述します。

- `configs/official.toml`: MetaDrive公式の Stable-Baselines3 サンプル相当の例です。
- `configs/generalization.toml`: 手続き生成道路で学習し、別の scenario 範囲で評価する設定です。
- `configs/example_experiment.toml`: 新しい実験を作るためのテンプレートです。
- `configs/official_start_lane_return.toml`: reset 時の開始レーンを維持して到着することを目指す baseline です。
- `configs/01_official_start_lane_return_idle_penalty.toml`: 低速 penalty の一要因実験です。
- `configs/02_official_start_lane_return_progress_balance.toml`: 開始レーン中心 cost の係数を下げる一要因実験です。
- `configs/03_official_start_lane_return_timeout_penalty.toml`: 純粋な時間切れ penalty の一要因実験です。
- `configs/04_official_start_lane_return_duckietown_progress.toml`: 開始レーン内の前進距離だけを報酬化する Duckietown 方式の実験です。

開始レーン維持実験の実行順、設定差、評価結果、成果物、再現方法は [実験記録](docs/official_start_lane_return_experiments.md) を参照してください。

テンプレートをコピーして値を編集し、学習と評価に同じファイルを渡します。

```bash
cp configs/example_experiment.toml configs/my_experiment.toml
.venv/bin/python train.py --config configs/my_experiment.toml
.venv/bin/python evaluate.py --config configs/my_experiment.toml
```

**root**

| 項目 | 必須 | 意味 |
| --- | --- | --- |
| `schema_version` | はい | 設定形式の識別子です。テンプレートの値を変更しません。 |
| `name` | はい | 実験設定の名前です。成果物の `outputs/<name>/` に使われます。 |
| `algorithm` | はい | 学習アルゴリズムです。`ppo` を指定します。 |
| `default_model_name` | いいえ | 学習モデルの標準 basename です。`training.model_name` を省略した場合に使われます。 |
| `[training]` | はい | PPO の学習設定です。 |
| `[evaluation]` | はい | 保存済みモデルの評価設定です。table 自体は必須で、内部 key はすべて任意です。 |
| `[environment]` | はい | MetaDrive 環境設定です。 |

**`[training]`**

| 項目 | 必須 | 意味 |
| --- | --- | --- |
| `policy` | はい | PPO に渡す Policy 名です。公式設定は `MlpPolicy` です。 |
| `seed` | はい | PPO と学習 worker の乱数 seed です。 |
| `num_envs` | はい | `SubprocVecEnv` で並列に動かす環境数です。 |
| `n_steps` | はい | PPO 更新前に各環境から収集する rollout 長です。 |
| `total_timesteps` | はい | `model.learn()` に要求する最小環境 step 数です。 |
| `log_interval` | はい | `model.learn()` のログ出力間隔です。 |
| `device` | いいえ | PPO に渡す計算デバイスです。例: `cpu`、`cuda`、`auto`。 |
| `model_name` | いいえ | `models/<model_name>.zip` と学習成果物ディレクトリに使う basename です。 |
| `log_file` | いいえ | 標準出力と標準エラーの複製先です。相対パスはプロジェクト直下から解決されます。 |
| `learning_rate` | いいえ | PPO optimizer の学習率です。 |
| `batch_size` | いいえ | PPO 更新時の mini-batch サイズです。 |
| `n_epochs` | いいえ | 1 rollout に対する PPO 更新 epoch 数です。 |
| `gamma` | いいえ | 割引率です。 |
| `gae_lambda` | いいえ | Generalized Advantage Estimation の係数です。 |
| `clip_range` | いいえ | PPO の方策更新を制限する clip 範囲です。 |
| `normalize_advantage` | いいえ | advantage を正規化するかどうかです。 |
| `ent_coef` | いいえ | entropy bonus の係数です。 |
| `vf_coef` | いいえ | value function loss の係数です。 |
| `max_grad_norm` | いいえ | 勾配 clipping の最大ノルムです。 |

`learning_rate` から `max_grad_norm` までは省略時に PPO の標準値が補完されます。

**`[evaluation]`**

| 項目 | 必須 | 意味 |
| --- | --- | --- |
| `model_path` | いいえ | 読み込む `.zip` モデルのパスです。省略時は学習モデル名から `models/` 配下を選びます。 |
| `record_gif` | いいえ | 各評価 episode の GIF、MP4、PNG を記録するかどうかです。 |
| `output_prefix` | いいえ | `outputs/<name>/evaluation/` とログに使う basename です。 |
| `seed` | いいえ | 評価時の乱数 seed です。省略時は `training.seed` を使います。 |
| `device` | いいえ | `PPO.load()` に渡す計算デバイスです。省略時は `cpu` です。 |
| `log_file` | いいえ | 標準出力と標準エラーの複製先です。相対パスはプロジェクト直下から解決されます。 |
| `deterministic` | いいえ | `model.predict()` で決定論的に Action を選ぶかどうかです。 |

**`[environment]`**

`[environment.common]` の値は `[environment.train]` と `[environment.evaluation]` に再帰的に統合され、同じ項目は各 stage 側の値が優先されます。各 stage の有効な設定には `start_seed` と `num_scenarios` が必要です。

| 項目 | 必須 | 意味 |
| --- | --- | --- |
| `[environment.common]` | いいえ | 学習と評価に共通の MetaDrive 設定です。 |
| `[environment.train]` | はい | 学習環境だけに適用する設定です。 |
| `[environment.evaluation]` | はい | 評価環境だけに適用する設定です。評価は指定した scenario 範囲を先頭から1回ずつ実行します。 |
| `start_seed` | はい（各 stage） | scenario 範囲の先頭 seed です。 |
| `num_scenarios` | はい（各 stage） | scenario の数です。正の整数を指定します。 |

公式設定で `[environment.common]` に置く主要な MetaDrive 項目は次のとおりです。

| 項目 | 公式設定の値 | 意味 |
| --- | --- | --- |
| `map` | `"C"` | 使用する道路 map です。 |
| `discrete_action` | `true` | 離散 Action を使います。 |
| `discrete_steering_dim` | `3` | steering の離散段階数です。 |
| `discrete_throttle_dim` | `3` | throttle/brake の離散段階数です。3×3 で9 Actionになります。 |
| `horizon` | `500` | 1 episode の最大 step 数です。 |
| `random_spawn_lane_index` | `false` | spawn lane をランダムに選ぶかどうかです。 |
| `traffic_density` | `0` | 交通量です。 |
| `accident_prob` | `0` | 事故生成の確率です。 |
| `log_level` | `50` | MetaDrive のログレベルです。 |

そのほかの MetaDrive 設定も `common` または各 stage に記述できます。評価経路は single-agent の離散 Action を対象とするため、`discrete_action = true` が必要です。指定する場合は `use_multi_discrete = false`、`is_multi_agent = false`、`num_agents = 1` とし、離散 Action の各次元は2以上にします。
