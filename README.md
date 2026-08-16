# MetaDrive公式SB3ミニ例 Phase 0

このディレクトリは、MetaDrive公式ドキュメントの「Training > stable-baselines3」にある最小構成を、学習タスクを変えずに通常のPythonスクリプトへ分割したものです。固定された `map="C"` の道路でMetaDrive標準Observationを受け取り、9種類の離散Actionから操作を選び、MetaDrive標準Rewardを最大化しながら目的地へ向かうPolicyをSB3 PPOで学習します。

> **2026-08-16 最終実測:** project-localのuv 0.12.5でCPython 3.11.16をプロジェクト内へ導入し、exact tag `MetaDrive-0.4.3` / commit `5bf8ea8909c4643a4099a250e6f5fb89c695d8b4`をinstallしました。assets取得、公式10,000-step profile、pytest 3件、Smoke学習・評価、公式設定の本学習311,296 timesteps、決定論的評価、600×600 GIF生成まで終了コード0です。本評価はreward `62.986104368858946`、length 68、目的地未到着、道路外終了でした。未達の字義条件は、公式設定のraw環境へSB3 2.9.0の`check_env`を直接適用した場合のfatal-free完走だけです。検査専用seed adapterではpassしましたが、raw直接検査の成功とは扱いません。実行済み事実と各終了コードは `RUN_REPORT.md` を正とします。

## 1. 今回のタスク

### 学習すること

- 道路とシナリオは `map="C"`、`num_scenarios=1`、`start_seed=5` に固定します。このtagのソースではブロックID `C` は `Curve` です。公式ミニ例と同じ対象を使い、複数マップやseedに対する一般化は扱いません。
- 交通量を0、事故生成確率を0にし、周辺車との交渉ではなく、自己車両の経路追従と目的地到達という最小の接続を検証します。
- `discrete_steering_dim=3` と `discrete_throttle_dim=3` の直積を単一の `Discrete(9)` とし、操作候補を小さくします。
- PPOと `MlpPolicy` を使います。これは公式ミニ例と同じ組合せであり、離散Actionを直接扱えて、MetaDriveとSB3の接続確認に必要十分だからです。

### 学習しないこと

他車回避、交通交渉、障害物回避、複数マップ・複数scenario seedへの一般化は対象外です。また、独自Observation、独自Reward、独自終了条件、独自Policyネットワーク、独自Feature Extractor、独自Controller、連続Action、画像Observation、マルチエージェント、SAC/TD3との比較、ハイパーパラメータ探索は導入しません。`use_lateral_reward=True` も追加しません。

`map="C"`、離散Action、PPOを採用する第一の理由は、公式例の学習問題そのものを再現し、後続研究の変更を混ぜる前に「MetaDrive → SB3 → PPO更新 → 保存モデルの評価」という配線を検証するためです。

## 2. 固定する契約

環境へ明示的に渡す設定は次の11項目だけです。Rewardや終了条件に関する設定は追加せず、確認したMetaDrive tagのデフォルトを使います。

```python
OFFICIAL_ENV_CONFIG = {
    "map": "C",
    "discrete_action": True,
    "discrete_throttle_dim": 3,
    "discrete_steering_dim": 3,
    "horizon": 500,
    "random_spawn_lane_index": False,
    "num_scenarios": 1,
    "start_seed": 5,
    "traffic_density": 0,
    "accident_prob": 0,
    "log_level": 50,
}
```

契約を短くまとめると次のとおりです。

| 項目 | Phase 0の値 |
| --- | --- |
| Observation | MetaDrive標準 |
| Action | `Discrete(9)` |
| Reward | MetaDrive標準 |
| 終了条件 | MetaDrive標準 |
| シナリオ | 1個、`start_seed=5` |
| 交通量 / 事故生成 | `0` / `0` |
| RL algorithm | SB3 PPO |
| Policy | `MlpPolicy` |

本学習の設定も公式例に合わせます。

```python
OFFICIAL_TRAINING_CONFIG = {
    "seed": 0,
    "num_envs": 4,
    "n_steps": 4096,
    "total_timesteps": 300_000,
    "log_interval": 4,
    "policy": "MlpPolicy",
}
```

PPOのその他のハイパーパラメータはSB3のデフォルトを使います。`device` だけは実行環境を曖昧にしないためCLIで指定し、既定を `cpu` とします。これはSB3がPPO + `MlpPolicy`にCPUを推奨していることに合わせた性能上の既定値であり、ホストGPUが存在しないという判定ではありません。CUDA対応PyTorch環境では`--device cuda`を明示してGPU学習も実行できます。

RL seedは公式例どおり `set_random_seed(0)` で設定し、`PPO(seed=...)` は指定しません。SB3 2.9.0でPPOへseedを渡すとVecEnvにもseedが転送されますが、MetaDrive 0.4.3の `reset(seed=...)` は通常の乱数seedではなくscenario indexとして扱われます。今回の有効なscenario indexは5だけなので、RL seed 0を環境resetへ流すと固定scenarioを壊して失敗します。Action/Observation spaceの乱数だけはworkerごとにseedし、環境resetは引数なしで行います。

### Action IDの意味

確認した `EnvInputPolicy.convert_to_continuous_action()` は、単一のAction IDを `id % 3` でsteering、`id // 3` でthrottle/brakeへ割り当てます。各軸は3段階なので、変換値は次のとおりです。符号へ「左」「右」の名称は付けず、ソースから直接確認できる数値だけを示します。

| Action ID | steering | throttle_brake |
| ---: | ---: | ---: |
| 0 | -1.0 | -1.0 |
| 1 | 0.0 | -1.0 |
| 2 | +1.0 | -1.0 |
| 3 | -1.0 | 0.0 |
| 4 | 0.0 | 0.0 |
| 5 | +1.0 | 0.0 |
| 6 | -1.0 | +1.0 |
| 7 | 0.0 | +1.0 |
| 8 | +1.0 | +1.0 |

## 3. MDPとの対応

| MDP要素 | 今回の担当 |
| --- | --- |
| State / Observation | MetaDrive標準Observation |
| Action | 9種類の離散Action |
| Transition | MetaDriveの車両物理・シミュレーション |
| Reward | MetaDrive標準Reward |
| Discount factor | SB3 PPOの `gamma` |
| Policy | SB3の `MlpPolicy` |
| Episode termination | MetaDriveの終了判定 |
| Timeout | `horizon=500` による `truncated` |

環境の完全な内部状態（State）と、Policyへ渡される観測（Observation）は厳密には同じとは限りません。このPhase 0ではMetaDriveが標準で生成するObservationだけをPolicyへ渡し、独自の状態表現は追加しません。`gamma` も自作せず、実際に導入できたSB3 2.9.0 PPOのデフォルト `0.99` を使います。

## 4. データフロー

```text
MetaDrive標準Observation
        ↓
SB3 PPOのActor
        ↓
離散Action ID 0～8
        ↓
MetaDrive EnvInputPolicy
        ↓
[steering, throttle_brake]
        ↓
MetaDrive車両物理
        ↓
次のObservation・Reward・terminated・truncated・info
        ↓
SB3のrollout buffer
        ↓
PPOによるActor/Critic更新
```

評価時はActorからActionを得てMetaDriveを進めるところまでは同じですが、rollout bufferへの学習用蓄積、逆伝播、重み更新は行いません。

## 5. 責任分担

| 処理 | 担当 |
| --- | --- |
| 道路生成 | MetaDrive |
| 車両物理 | MetaDrive |
| Observation生成 | MetaDrive |
| Reward計算 | MetaDrive |
| 終了判定 | MetaDrive |
| 離散Actionから連続制御への変換 | MetaDrive `EnvInputPolicy` |
| rollout収集の制御 | SB3 |
| Actor/Critic | SB3 |
| 損失計算 | SB3 |
| 逆伝播・重み更新 | PyTorch/SB3 |
| 環境の並列化 | SB3 `SubprocVecEnv` |
| エピソードログ | SB3 `Monitor` |
| 学習開始指示 | 自作 `train.py` |
| 保存モデルによる推論 | 自作 `evaluate.py` ＋ SB3 |

自作wrapperの `Monitor` は記録だけを担当し、Observation、Action、Reward、終了条件を変更しません。4個の学習環境はMetaDriveの「1プロセスにつき1環境」という制約を守るため、`SubprocVecEnv` で別プロセスへ1個ずつ配置します。

## 6. MetaDrive標準Reward

以下は、checkout・editable install済みのcommit `5bf8ea8909c4643a4099a250e6f5fb89c695d8b4` にある `metadrive/envs/metadrive_env.py` の実際の設定と `MetaDriveEnv.reward_function()`、およびPython 3.11.16上の実 `env.config` の確認結果に基づきます。

| 要素 | このtagのデフォルト | 内容 |
| --- | ---: | --- |
| 前進Reward | `driving_reward=1.0` | 現在lane上の縦方向進捗 `long_now - long_last` に掛ける |
| 速度Reward | `speed_reward=0.1` | `speed_km_h / max_speed_km_h` に掛ける |
| 到着Reward | `success_reward=10.0` | 到着stepの最終Rewardへ置換 |
| 道路外Penalty | `out_of_road_penalty=5.0` | 道路外stepの最終Rewardを `-5.0` へ置換 |
| 車両衝突Penalty | `crash_vehicle_penalty=5.0` | 車両衝突stepの最終Rewardを `-5.0` へ置換 |
| 物体衝突Penalty | `crash_object_penalty=5.0` | 物体衝突stepの最終Rewardを `-5.0` へ置換 |
| 横方向係数 | `use_lateral_reward=False` | Phase 0では上書きしないため係数は常に `1.0` |

通常stepのdense rewardは概念的に次の和です。`positive_road` は道路方向に応じて `+1` または `-1` になります。

```text
driving_reward × 縦方向進捗 × lateral_factor × positive_road
+ speed_reward × 正規化速度 × positive_road
```

`use_lateral_reward=True` のときだけlane中心からの横ずれに応じて `lateral_factor` が0～1へ縮みます。このtagの実際のデフォルトは `False` であり、Phase 0では指定にも追加しません。

重要なのは終了stepの扱いです。関数はいったんdense rewardを `step_info["step_reward"]` に保存した後、到着、道路外、車両衝突、物体衝突、歩道衝突の優先順で、返却するRewardをterminal reward/penaltyへ**置き換えます**。つまり、到着時にdense rewardへ `+10` を加える実装ではありません。複数フラグが同時に立つ可能性があっても、評価JSONには元フラグをすべて残します。

## 7. `terminated` と `truncated`

`terminated` は到着、道路外、衝突など、MetaDriveが定義するタスク上の終了です。このtagの標準終了判定には、到着、道路外（lane外や設定対象の連続線・歩道衝突を含む）、車両・物体・建物・人との衝突などが含まれます。Phase 0ではその標準設定を変更しません。

`truncated` は時間制限による打切りです。今回は `horizon=500` なので、エピソード長が500stepに達すると `max_step=True` と `truncated=True` になります。確認した `BaseEnv` の実際のデフォルトは `truncate_as_terminate=False` です。そのため、時間切れだけなら `terminated=False` のままです。タスク終了と時間切れが同じstepで発生すれば、両方が `True` になることもあります。

評価ループを `terminated or truncated` で止めるのは、どちらもそのエピソードから先へ `step()` してはいけない境界だからです。ただし、結果では両者を混ぜず、終了フラグと `success`、`out_of_road`、`crash_vehicle`、`crash_object`、`other_termination`、`max_step_truncation`、`unknown` の理由を区別します。

## 8. rollout単位と実timesteps

SB3のon-policy学習は、`n_steps × num_envs` 件を1 rolloutとして収集してからPPO更新を行います。`total_timesteps` は厳密な停止位置ではなく下限なので、最後のrolloutを途中で切らず、指定値を超えることがあります。

- 本学習: `4096 × 4 = 16,384` step/rollout。`ceil(300,000 / 16,384) = 19` rolloutで、実際に **311,296** stepを収集して終了コード0で完了しました。
- Smoke Test: `256 × 1 = 256` step/rollout。`ceil(2,000 / 256) = 8` rolloutで、実際に **2,048** stepを収集して終了コード0で完了しました。

TensorBoardの最終scalar stepが262,144なのは学習不足ではありません。`log_interval=4`では4 rolloutごとに記録するため、19 rolloutの最後の記録点は16 rollout完了時の`16,384 × 16 = 262,144`です。学習metadataが記録した最終`actual_total_timesteps=311296`を実収集量とします。

## 9. 公式例から変更したもの

| 項目 | 公式例 | 今回 |
| --- | --- | --- |
| 学習タスク | 公式設定 | 同じ |
| Observation | 標準 | 同じ |
| Reward | 標準 | 同じ |
| Action | `Discrete(9)` | 同じ |
| PPO | `MlpPolicy` | 同じ |
| 本学習 | 4環境、300,000 step | 同じ |
| コード構成 | Notebook的な一連の例 | 複数ファイルへ分割 |
| 環境検査 | なし | `check_env` 等を追加 |
| Smoke Test | `TEST_DOC` 時の簡略実行 | 明示的な短時間設定 |
| 終了判定 | サンプルでは簡略化 | `terminated` と `truncated` を区別 |
| バージョン記録 | 最小限 | tag・commit・package versionを保存 |
| device | 例の実行環境へ委任 | CLIで明示、既定 `cpu`、CUDA検証時は`cuda` |
| GIF出力先 | `demo.gif` | `outputs/` へ整理 |

変更はコード分割、検査、記録、CLI、出力整理に限られます。Observation、Action、Reward、終了条件、シナリオという学習タスクは変更しません。

## 10. ソースとバージョンの確認結果

一次資料として公式ドキュメントと公式Git repositoryを使い、実際にcheckoutしたソースを優先します。

- Git remote: `https://github.com/metadriverse/metadrive.git`
- release tag: `MetaDrive-0.4.3`
- tag/checkout commit: `5bf8ea8909c4643a4099a250e6f5fb89c695d8b4`
- checkout状態: detached HEAD、作業tree変更なし（確認時）
- `metadrive/version.py` の `VERSION`: `0.4.2.3`
- `setup.py` のPython要件: `>=3.6, <3.12`

tag名は0.4.3ですが、同じcommit内のdistribution、`metadrive.version.VERSION`、取得assetsはすべて `0.4.2.3` です。packageには`metadrive.__version__`属性がありません。この不一致をこちらで書き換えたり、`0.4.3` がimport結果になると推測したりしません。`environment_report.json` ではGit tag、commit、distribution/source/assets versionを別フィールドに記録しています。

2026-08-16のCPU venvで実測した主なversionは、Python 3.11.16、SB3 2.9.0、Gymnasium 1.3.0、PyTorch 2.13.0+cpu、NumPy 2.4.6、Panda3D 1.10.13です。このvenvは`--torch-backend cpu`でCPU版PyTorchを明示的に導入し、学習にも`--device cpu`を指定したため、`torch.cuda.is_available()`は`False`でした。この値はCPU wheelを選んだ結果であり、ホストGPUが存在しないことの証拠ではありません。公式`profile_metadrive`は10,000 simulation stepsを完走し、平均489.104 FPSでした。

2026-08-17にホスト側で再確認すると、`nvidia-smi`はNVIDIA GeForce RTX 3090 Ti、driver 610.88、VRAM 24,564 MiBを正常に認識しました。別の`.venv-cuda`を構築するとPyTorch 2.13.0+cu132、`torch.cuda.is_available() == True`となり、CUDA tensor演算、MetaDrive/SB3の2,048-step Smoke学習、保存modelのCUDA再読込、1 episode評価まで成功しました。詳細は`outputs/cuda_environment_retry_20260817.json`に分離して記録しています。

raw環境への `check_env` にはversion間のseed契約衝突があります。実インストール済みSB3 2.9.0のcheckerは `env.reset(seed=0)` を呼びますが、MetaDrive側はその値を乱数seedではなくscenario indexとして扱い、許容範囲は `[5, 6)` です。raw直接検査は実際に `AssertionError: scenario_index (seed) should be in [5:6)` で終了コード1となり、`outputs/inspect_env_raw_py311.log`へ保存しました。

最終`inspect_env.py`はraw failureを隠さず報告した後、検査時だけscenario 5を維持するseed adapterでも`check_env`を行い、こちらはpassしました。adapterはspaceをseedしつつraw envをscenario 5でresetする検査専用処理で、公式11設定を変えず、学習・評価には使いません。そのため、adapter成功をraw環境のfatal-free完走とは数えません。詳しいsource行は `CODE_WALKTHROUGH.md` に記録しています。

参照先:

- [MetaDrive Training](https://metadrive-simulator.readthedocs.io/en/latest/training.html)
- [MetaDrive Action and Policy](https://metadrive-simulator.readthedocs.io/en/latest/action.html)
- [MetaDrive Reward, Cost, Termination, and Step Information](https://metadrive-simulator.readthedocs.io/en/latest/reward_cost_done.html)
- [MetaDrive Installing MetaDrive](https://metadrive-simulator.readthedocs.io/en/latest/install.html)
- [MetaDrive公式repository](https://github.com/metadriverse/metadrive)
- [SB3 Custom Environments](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)

## 11. ファイル構成

```text
metadrive_sb3_phase0/
├── .gitignore                       # venv、upstream clone、モデル、ログ、大きな描画物を除外
├── requirements.txt                 # 直接依存（editable MetaDrive、SB3、pytest）
├── requirements.lock.txt            # 2026-08-16 CPU venvのpip freeze
├── requirements.cuda.lock.txt       # 2026-08-17 CUDA venvのpip freeze
├── phase0_config.py                  # 公式設定値と出力先の一元管理
├── env_factory.py                    # MetaDrive生成と記録専用Monitor
├── inspect_env.py                    # version・空間・Action変換・check_env・random走行検査
├── train.py                          # SubprocVecEnvとPPO学習・モデル保存
├── evaluate.py                       # 保存モデルの決定論的評価と任意のGIF記録
├── README.md                         # タスク、設計、実行手順（本書）
├── CODE_WALKTHROUGH.md               # 自作・library内部処理の行番号付き解説
├── RUN_REPORT.md                     # 実際に実行した結果、終了コード、未解決事項
├── tests/test_phase0_contract.py     # 重い学習を含めない環境契約テスト
├── models/                           # phase0_smoke.zip / phase0_official.zip
├── logs/
│   ├── monitor/                      # rank別のMonitor CSV
│   └── tensorboard/                  # TensorBoard event
├── outputs/                          # 検査log、metadata、評価JSON、GIF、CPU/CUDA環境report
└── .external/metadrive-src/          # 公式tagのeditable checkout
```

`.venv/`、`.venv-cuda/` と `.external/` はlocal再構築物なのでgit管理外です。モデル、TensorBoard/Monitorログ、GIF・動画・画像も容量が大きくなり得るため除外します。一方、CPU/CUDA両方のlock、JSON report、README類、空ディレクトリを保つ `.gitkeep` は追跡可能にします。

## 12. 環境構築

すべてのコマンドはこのディレクトリをcurrent directoryとして実行し、activationには依存しません。

```bash
cd /home/seigo/workspace/metadrive_rl/metadrive_sb3_phase0
```

### 12.1 Python 3.12で失敗した履歴

初回確認時のsystem PythonはUbuntu 24.04.4 LTS on WSL2の`/usr/bin/python3` = Python 3.12.3で、`python`コマンドはありませんでした。指示どおりPython 3.12で最初のvenvを作りましたが、次のMetaDrive installはsource内のassertで終了コード1になりました。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .external/metadrive-src
```

失敗は依存解決の偶発的なwarningではなく、このrelease tagが明記するPython上限との非互換です。全文は`outputs/metadrive_install_error.log`、当時のvenvは`.venv-py312-blocked/`へ保持しました。`setup.py`のassert削除、Python要件の改変、`sudo apt install`、system Pythonへの直接installは行っていません。

### 12.2 project-local uvとPython 3.11によるCPU環境構築（2026-08-16実績）

system packageやshell profileを変更せず、project-localの`.tools/uv` 0.12.5でCPython 3.11.16を`.uv-python/`へ導入しました。次は2026-08-16にCPU環境として実際に成功したオプションを含む再現コマンドです。`--no-bin`はmanaged Pythonへのglobal shim追加を避け、`--managed-python`はuv管理interpreterを選びます。activationは不要です。

```bash
curl -LsSf \
  https://releases.astral.sh/github/uv/releases/download/0.12.5/uv-installer.sh \
  -o /tmp/metadrive-uv-0.12.5-installer.sh
printf '%s  %s\n' \
  504511fbbbd811aeaba6738abc79408956b6c7da0ca35437b3dcc24a41efc111 \
  /tmp/metadrive-uv-0.12.5-installer.sh | sha256sum -c -
env UV_UNMANAGED_INSTALL="$PWD/.tools" \
  /bin/sh /tmp/metadrive-uv-0.12.5-installer.sh
./.tools/uv --version
env UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" UV_CACHE_DIR="$PWD/.uv-cache" \
  ./.tools/uv python install 3.11 --no-bin
env UV_PYTHON_INSTALL_DIR="$PWD/.uv-python" UV_CACHE_DIR="$PWD/.uv-cache" \
  ./.tools/uv venv --python 3.11 --managed-python --seed .venv
.venv/bin/python --version
```

`UV_UNMANAGED_INSTALL`によりuv本体もproject内へ置き、installerによるshell
profile変更とself-updateを無効にしています。この実行で取得したinstallerの
SHA-256は`504511fbbbd811aeaba6738abc79408956b6c7da0ca35437b3dcc24a41efc111`、
導入されたuvは0.12.5でした。

公式tagとcommitは次の順で取得・検証します。すでに正しいcheckoutがある場合、再cloneは不要です。

```bash
git ls-remote --tags https://github.com/metadriverse/metadrive.git refs/tags/MetaDrive-0.4.3
git clone https://github.com/metadriverse/metadrive.git .external/metadrive-src
git -C .external/metadrive-src checkout --detach MetaDrive-0.4.3
git -C .external/metadrive-src rev-parse HEAD
git -C .external/metadrive-src describe --tags --exact-match
```

`rev-parse HEAD` が `5bf8ea8909c4643a4099a250e6f5fb89c695d8b4`、`describe` が `MetaDrive-0.4.3` であることを確認してから依存関係を導入します。`requirements.txt`のeditable pathはこの検証済みcheckoutを指します。この履歴では`--torch-backend cpu`を明示してCPU wheel経路を選び、`--strict`で解決時の不整合を許容しません。

```bash
UV_CACHE_DIR="$PWD/.uv-cache" ./.tools/uv pip install \
  --python .venv/bin/python \
  --torch-backend cpu \
  --strict \
  -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > requirements.lock.txt
```

`requirements.lock.txt`は、MetaDrive、Panda3D、SB3、Gymnasium、CPU版PyTorchを含むこのCPU venvから実行した`pip freeze`の実測76行です。SHA-256は`052b12473970e1efebf22d3a86f82435cb564847f674f44989cc38cccb1bd860`です。`pip check`も終了コード0で`No broken requirements found`でした。

### 12.3 CUDA環境の非破壊再構築（2026-08-17実績）

ホストGPUの有無とCPU版PyTorchの状態を混同しないため、既存`.venv`を残したまま`.venv-cuda`を作りました。GPUが見えるホスト実行コンテキストで`--torch-backend auto`を使うと、uv 0.12.5はPyTorch 2.13.0+cu132を解決しました。再現時は同じbackendを明示し、PyTorchのversionも固定します。

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
  --format=csv,noheader

./.tools/uv venv \
  --cache-dir .uv-cache \
  --python .uv-python/cpython-3.11.16-linux-x86_64-gnu/bin/python3.11 \
  --seed \
  .venv-cuda

./.tools/uv pip install \
  --cache-dir .uv-cache \
  --python .venv-cuda/bin/python \
  --torch-backend cu132 \
  --strict \
  -r requirements.txt \
  'torch==2.13.0'

.venv-cuda/bin/python -m pip check
.venv-cuda/bin/python -m pytest -q
```

実測結果は次のとおりです。

| 項目 | 実測値 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 Ti、24,564 MiB、compute capability 8.6 |
| driver | 610.88 |
| PyTorch | 2.13.0+cu132 |
| `torch.version.cuda` | 13.2 |
| `torch.cuda.is_available()` | `True` |
| CUDA device | `cuda:0` / NVIDIA GeForce RTX 3090 Ti |
| CUDA probe | 1024×1024 matrix multiplication、同期、有限値検査に成功 |
| `pip check` | `No broken requirements found` |
| pytest | 3 passed in 8.09s |

CUDA venvの`pip freeze`は`requirements.cuda.lock.txt`へ分離しました。95行、SHA-256は`55e40071d5aca3f48d896a933120be076af3ed8612c288bc93011238720cf877`です。CPU実績の`requirements.lock.txt`と`outputs/environment_report.json`は当時の履歴として変更せず、CUDA再構築の機械可読な記録は`outputs/cuda_environment_retry_20260817.json`へ保存します。

## 13. 実行手順

CPU実績を再現する場合は12.2、CUDA環境を再構築する場合は12.3の手順を使います。以降もactivationに依存せず、`MPLCONFIGDIR`と`XDG_CACHE_HOME`をproject内へinline指定します。Python側でcanonical logを保存する`inspect_env.py`、`train.py`、`evaluate.py`には、**同じlogを開く外部`tee`を付けません**。同一ファイルを内外から同時にtruncateする競合を避けるためです。外部`tee`は内部logを持たないassets/profile/pytestだけに使い、`pipefail`でPythonの失敗を隠しません。

```bash
mkdir -p .runtime/config/matplotlib .runtime/cache
```

### 3. assets取得と公式install検証

package直下にある `metadrive` ディレクトリとの名前衝突を避けるため、安全な `outputs/` からmoduleを実行します。

```bash
set -o pipefail
(cd outputs && \
  env MPLCONFIGDIR="../.runtime/config/matplotlib" XDG_CACHE_HOME="../.runtime/cache" \
  ../.venv/bin/python -m metadrive.pull_asset) \
  2>&1 | tee outputs/pull_asset.log
(cd outputs && \
  env MPLCONFIGDIR="../.runtime/config/matplotlib" XDG_CACHE_HOME="../.runtime/cache" \
  ../.venv/bin/python -m metadrive.examples.profile_metadrive) \
  2>&1 | tee outputs/profile_metadrive.log
```

最終実測ではassets 0.4.2.3を取得し、profileは10,000 simulation steps、平均489.104 FPSで終了コード0でした。元の証跡は`outputs/pull_asset_py311.log`と`outputs/profile_metadrive_py311.log`です。

### 4. `inspect_env`

version、適用configと実 `env.config`、Observation/Action space、Observationのshape/dtype/range/有限性、reset/step info、9 Actionの内部変換、Gymnasium 5戻り値、`check_env`、最大50stepのrandom走行を確認します。warningは隠しません。

```bash
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv/bin/python inspect_env.py
```

この最終commandの終了コードは0です。ただし、内部ではraw直接`check_env`の既知のfatalを`outputs/inspect_env.log`へ明示し、検査専用adapterのpassと50step走行まで確認します。raw検査だけを実行した履歴の終了コードは1で、`outputs/inspect_env_raw_py311.log`へ保存しています。

### 5. pytest

環境生成、`Discrete(9)`、Observation妥当性、reset/step API、Rewardと終了flagの型、公式設定不変、closeを検査します。300,000 step学習はtestへ含めません。

```bash
set -o pipefail
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv/bin/python -m pytest -q 2>&1 | tee outputs/pytest.log
```

最終結果は3 passed in 4.28s、終了コード0です。

### 6. Smoke Test

これは公式本学習ではなく、PPO初期化、rollout収集、勾配更新、保存・再読込までの配線確認です。`num_envs=1`、`n_steps=256`、`timesteps=2000` は本学習の公式値と異なります。

```bash
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv/bin/python train.py \
  --timesteps 2000 \
  --num-envs 1 \
  --n-steps 256 \
  --seed 0 \
  --device cpu \
  --model-name phase0_smoke
```

script自身が`logs/smoke_train.log`へ保存します。実際にはrollout単位で2,048 timestepsを学習し、model保存・再読込まで終了コード0でした。

### 7. Smokeモデル評価

```bash
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv/bin/python evaluate.py \
  --model models/phase0_smoke.zip \
  --episodes 1 \
  --output-prefix phase0_smoke
```

script自身が`logs/evaluate_smoke.log`へ保存します。1 episode評価は終了コード0でした。

本学習はrank 0のcanonical Monitor名を再利用するため、SmokeのMonitorを
保持する場合は、本学習前に今回と同じく退避します。

```bash
cp logs/monitor/env_0.monitor.csv logs/monitor/smoke_env_0.monitor.csv
```

同じ`--model-name`や`--output-prefix`で再実行すると、対応するmodel、metadata
JSON、console log、Monitor、評価JSON、GIFは上書きされます。過去runを保持する
場合は、事前に退避するか別名を指定してください。

### 7.1 CUDA Smoke学習・評価（2026-08-17再試行）

CUDA経路は既存CPU成果物と衝突しない名前で検証しました。学習前に最新100万step実験のrank 0 Monitorを`phase0_1m_env_0.monitor.csv`へ退避し、CUDA Smoke後のMonitorを`cuda_smoke_env_0.monitor.csv`へ保存してからcanonical `env_0.monitor.csv`を元のSHA-256へ復元しています。

```bash
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv-cuda/bin/python train.py \
  --timesteps 2000 \
  --num-envs 1 \
  --n-steps 256 \
  --seed 0 \
  --device cuda \
  --model-name phase0_cuda_smoke \
  --log-file logs/cuda_smoke_train.log

env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv-cuda/bin/python evaluate.py \
  --model models/phase0_cuda_smoke.zip \
  --episodes 1 \
  --device cuda \
  --output-prefix phase0_cuda_smoke \
  --log-file logs/cuda_smoke_evaluate.log
```

学習は要求2,000に対して2,048 timestepsを完走し、`requested_device=cuda`、`actual_device=cuda`、保存modelの再読込deviceも`cuda`でした。評価も`requested_device=cuda`、`actual_device=cuda`で終了コード0です。SB3はPPO + `MlpPolicy`についてGPU利用率が低くCPUより遅くなり得るというwarningを出しますが、これは性能上の推奨であり、CUDAが動作していないという意味ではありません。

### 8. 公式本学習

```bash
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv/bin/python train.py \
  --timesteps 300000 \
  --num-envs 4 \
  --n-steps 4096 \
  --seed 0 \
  --device cpu \
  --model-name phase0_official
```

script自身が`logs/full_train.log`へ保存します。実際には19 rollout、311,296 timestepsを学習し、保存・再読込まで終了コード0でした。

### 9. 本学習モデル評価とGIF

```bash
env MPLCONFIGDIR="$PWD/.runtime/config/matplotlib" \
  XDG_CACHE_HOME="$PWD/.runtime/cache" \
  .venv/bin/python evaluate.py \
  --model models/phase0_official.zip \
  --episodes 1 \
  --record-gif \
  --output-prefix phase0_official
```

script自身が`logs/evaluate_official.log`へ保存します。1 episode評価とGIF生成は終了コード0でした。

### 10. TensorBoard

```bash
.venv/bin/python -m tensorboard.main --logdir logs/tensorboard
```

## 14. 評価とGIF API

評価は単一環境で、保存済みPPOを `deterministic=True` で呼び出します。各エピソードは `terminated or truncated` で終了し、total reward、length、元の終了flag、`route_completion`、model SHA-256、実行時間をJSONへ保存します。`info` のoptional項目は `info.get(...)` で取得し、存在しないキーを成功・失敗どちらにも推測しません。

このcommitで確認できるtop-down記録APIは次の形です。

```python
env.render(
    mode="topdown",
    window=False,
    screen_record=True,
)
env.top_down_renderer.generate_gif(
    gif_name="outputs/phase0_official_evaluation.gif",
    duration=30,
)
```

`window=False` はoff-screen描画、`screen_record=True` は各render frameの保存を意味します。`TopDownRenderer.generate_gif` の実際のsignatureは `(gif_name="demo.gif", duration=30)` で、`duration` の単位はmsです。ソース中の説明には第2引数を `fps` と呼ぶ古い表記もありますが、実signatureを優先します。rendererは最初のtop-down render後に利用します。互換処理はこのAPI差分だけを扱い、シミュレーション設定は変えません。

実際のGIF生成は終了コード0で、`outputs/phase0_official_evaluation.gif`を環境close前に検証しました。size 29,630 bytes、600×600、SHA-256 `0a1f661f201f3c49dce80e0f4488647d490d7dbe8f1493fa7c01e3686e6aa334`です。episodeは68 decision framesですが、Pillowが同一の連続frameをまとめるためencoded frame数は66です。duration合計は2,040 msで、68 × 30 msと一致します。

## 15. 主な成果物

次の成果物はすべて存在と非zero sizeを確認済みです。

| 種類 | 保存先 |
| --- | --- |
| Smoke model | `models/phase0_smoke.zip` |
| CUDA Smoke model | `models/phase0_cuda_smoke.zip` |
| 本学習model | `models/phase0_official.zip` |
| rank別Monitor | `logs/monitor/env_0.monitor.csv` ～ `env_3.monitor.csv` |
| Smoke Monitor archive | `logs/monitor/smoke_env_0.monitor.csv` |
| CUDA Smoke Monitor archive | `logs/monitor/cuda_smoke_env_0.monitor.csv` |
| TensorBoard | `logs/tensorboard/` |
| Smoke / CUDA Smoke / 本学習log | `logs/smoke_train.log` / `logs/cuda_smoke_train.log` / `logs/full_train.log` |
| training metadata | `outputs/<model-name>_training_metadata.json` |
| Smoke評価 | `outputs/phase0_smoke_evaluation.json` |
| CUDA Smoke評価 | `outputs/phase0_cuda_smoke_evaluation.json` |
| 本学習評価 | `outputs/phase0_official_evaluation.json` |
| top-down GIF | `outputs/phase0_official_evaluation.gif` |
| CPU環境情報 | `outputs/environment_report.json` |
| CUDA再構築情報 | `outputs/cuda_environment_retry_20260817.json` |
| install/検査/test log | `outputs/*.log` |
| CPU / CUDA dependency lock | `requirements.lock.txt` / `requirements.cuda.lock.txt` |

### 実測結果

| 実行 | requested / actual | elapsed | 結果 |
| --- | --- | --- | --- |
| Smoke学習 | 2,000 / **2,048** | metadata 12.710秒 | model保存・再読込成功 |
| CUDA Smoke学習 | 2,000 / **2,048** | metadata 24.618秒 | CUDA学習、model保存・CUDA再読込成功 |
| 公式本学習 | 300,000 / **311,296** | metadata 446.234秒 / external wall 7:11.81 | model保存・再読込成功 |
| Smoke評価 | 1 episode | JSON全体2.246秒 | reward 62.986104、length 68、`out_of_road` |
| CUDA Smoke評価 | 1 episode | JSON全体2.835秒 | CUDA読込、reward 62.986104、length 68、`out_of_road` |
| 本学習model評価 | 1 episode | JSON全体5.392秒 / external wall 0:09.13 | reward 62.986104、length 68、`out_of_road`、GIF成功 |

本学習metadataの`elapsed_seconds=446.234`、metadata内UTC timestamp差約430.400秒、外部`time`のwall 431.81秒は別々の記録値として保持し、同じ値だったことにはしていません。TensorBoard最終scalar step 262,144は`log_interval=4`による記録間隔の結果で、実収集量311,296と矛盾しません。

両評価のepisode値は同一でしたが、model zipのSHA-256はSmoke `94033992c91c404a6633c9d21b534d64019518c384d699df799d7e2910672576`、公式model `5922bdeb0ffc21533b3abe02f7fd696397f8472dbd014532c19ba493314a7268`で異なります。1固定scenarioの1 deterministic episodeだけなので、この一致を同一modelや一般性能の証拠にはしません。目的地には到着しておらず、両方とも`arrive_dest=false`、`out_of_road=true`、`crash=true`、`crash_vehicle=false`、`crash_object=false`、`max_step=false`、`terminated=true`、`truncated=false`でした。

詳細な数値、全主要commandの終了コード、warning、失敗履歴は`RUN_REPORT.md`にまとめています。2026-08-16のCPU実績は`outputs/environment_report.json`、2026-08-17のCUDA再構築とSmoke実績は`outputs/cuda_environment_retry_20260817.json`に機械可読形式で保存しています。

## 16. 既知の制約

学習・評価・GIF生成を阻害する問題は残っていません。字義上ただ1つ未達なのは、公式11設定のraw `MetaDriveEnv`へSB3 2.9.0の`check_env`を直接適用し、fatalなしで完走する条件です。SB3が送る`reset(seed=0)`をMetaDriveがscenario indexとして解釈し、固定scenario範囲`[5, 6)`から外れるためです。

検査専用seed adapterを介した`check_env`、50step random走行、pytest 3件はすべてpassしていますが、adapter成功をraw直接検査成功とは扱いません。raw failureは`outputs/inspect_env_raw_py311.log`へ残しています。公式config、upstream source、SB3 checkerを都合よく改変することはPhase 0の範囲外です。

Python 3.12.3でのinstall failureも履歴として保持していますが、project-local CPython 3.11.16で解決し、その後の全実行は完了しました。CPU環境の再構築は12.2、CUDA環境は12.3、学習・評価だけを繰り返す場合は13章の該当手順から進めます。

実freezeには、MetaDriveが要求する`pygame==2.6.1`とSB3の`extra`が要求する
`pygame-ce==2.5.8`が共存します。両distributionは`pygame` module namespaceを
共有し、この実環境でimportされる実装はpygame-ce 2.5.8でした。profile、学習、
評価はすべて成功し`pip check`もpassしていますが、将来の再解決ではinstall順に
依存する潜在的なnamespace競合としてversionとimport実体を確認してください。
