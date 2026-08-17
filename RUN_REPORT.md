# MetaDrive RL 現行環境レポート

最終確認日: 2026-08-18（Asia/Tokyo）

## 構成

MetaDrive公式sourceとRL projectは、同じ親directoryに配置する。

```text
metadrive-workspace/
├── metadrive/       # MetaDrive公式source
└── metadrive-rl/    # SB3 projectとPython仮想環境
    └── .venv/
```

MetaDrive checkoutの確認値は次のとおりである。

| 項目 | 値 |
| --- | --- |
| remote | `https://github.com/metadriverse/metadrive.git` |
| branch | `main` |
| commit | `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` |
| install形式 | sibling source `../metadrive`からのeditable install |
| import先 | `../metadrive/metadrive/__init__.py` |

## pipによる環境構築

`requirements.txt`に`-e ../metadrive`が含まれているため、projectの依存packageとMetaDrive公式sourceを一度にinstallできる。

```bash
mkdir -p metadrive-workspace
cd metadrive-workspace
git clone https://github.com/SeigoKaji/metadrive_rl.git metadrive-rl
git clone --branch main https://github.com/metadriverse/metadrive.git metadrive

cd metadrive-rl
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m metadrive.pull_asset
```

依存定義は次の2ファイルで管理する。

| ファイル | 用途 |
| --- | --- |
| `requirements.txt` | MetaDrive、Stable-Baselines3、pytestの直接依存 |
| `requirements.lock.txt` | 検証済みPython 3.12環境の完全なversion固定 |

完全に固定された依存versionを使う場合は、次のようにinstallする。

```bash
.venv/bin/python -m pip install -r requirements.lock.txt
```

## 現行version

| 項目 | 実測値 |
| --- | --- |
| Python | 3.12.3 |
| pip | 26.2.1 |
| MetaDrive | 0.4.3（editable: `../metadrive`） |
| MetaDrive assets | 0.4.3 |
| Stable-Baselines3 | 2.9.0 |
| Gymnasium | 1.3.0 |
| PyTorch | 2.13.0+cu130 |
| NumPy | 2.5.2 |
| Panda3D | 1.10.16 |

この実行コンテキストでは`torch.cuda.is_available()`は`False`である。

## 標準コマンド

project directoryからPython仮想環境のinterpreterを直接実行する。

```bash
cd metadrive-workspace/metadrive-rl

# 環境検査
.venv/bin/python inspect_env.py

# test
.venv/bin/python -m pytest -q

# 学習
.venv/bin/python train.py

# 推論・評価
.venv/bin/python evaluate.py --model models/phase0_official.zip --episodes 1
```

## 検証結果

| 検証 | 結果 |
| --- | --- |
| `.venv/bin/python -m pip check` | exit 0、依存関係の破損なし |
| `.venv/bin/python -m metadrive.pull_asset` | exit 0、assets 0.4.3を取得 |
| `.venv/bin/python -m pytest -q` | exit 0、10 passed in 4.07s |
| generalization専用test | exit 0、7 passed in 3.59s |
| generalization E2E Smoke | 4 workerで256 timestep学習、保存・再読込、未見scenario 0〜4の評価とJSON生成が完了 |
| `.venv/bin/python inspect_env.py` | exit 0、検査adapterと走行検査が完了 |
| 明示的に`env.close()`する1,000-step simulation | exit 0、1,000 steps完了 |
| 追加環境変数なしのPPO 64-step学習・推論 | exit 0、rollout、parameter更新、`predict()`、1 step、環境closeが完了 |

SB3のraw `check_env`は`reset(seed=0)`を呼ぶ。一方、このprojectのMetaDrive環境ではseedをscenario indexとして扱い、許容範囲が`[5, 6)`であるため、raw環境への直接検査では次の不整合が発生する。

```text
AssertionError: scenario_index (seed) should be in [5:6)
```

`inspect_env.py`はこのraw環境の挙動を記録したうえで、scenario 5を維持する検査専用adapterを通して残りの検査を完了する。このadapterは学習環境と評価環境には使用しない。

upstreamの`python -m metadrive.examples.profile_metadrive`は10,000 stepsを完走し、平均475.160 FPSを記録したが、process終了時はexit 139だった。upstream exampleは`env.close()`を呼ばない。明示的に環境をcloseする1,000-step simulationとPPO 64-step学習はいずれもexit 0で完了している。
