# 別 PC への移植

## コピー範囲

移植用 ZIP は `input_attribution/` フォルダだけを含めます。文書、schema、解析用 config、任意依存 `requirements-ig.txt`、PORTABLE_FILES はすべてそのフォルダ内にあります。repo-root の `docs/`、`requirements.txt`、PORTABLE_FILES、学習用 `configs/` は収録しません。学習済みモデル、実験 output、venv、pip cache、font、MetaDrive の第三者 checkout も含めません。移植先リポジトリへ既存ファイルを上書きコピーせず、追加 package と文書を専用ディレクトリへ展開してください。

```bash
python -m input_attribution.pack --source-root . --output /tmp/input_attribution_portable.zip
unzip -l /tmp/input_attribution_portable.zip
```

ZIP は `input_attribution/` 配下だけを entry 順、DOS timestamp、ファイル mode 固定で収録し、同じ source bytes から同じ archive bytes を作れるようにしています。実験結果の `.jsonl`/`.npz`/動画、model archive、cache、font は除外されます。除外対象を一つでも必要とする場合は、ZIPへ混ぜずに移植先の実行環境から別途指定します。

run の stage が再実行された場合は `status.json` の `analysis_id`/`relative_dir` が指す最新の成功結果だけを report が読みます。過去の analysis ID を同じ表へ混ぜません。レポート再生成も root の初版を残し、`reports/<report_id>/` に保存します。

## 最小の編集箇所

移植先で確認・編集するのは解析用 TOML、adapter、schema の3箇所です。公式環境接続が使える PC では、同梱の `input_attribution/adapters/metadrive.py` を既存の環境 factory/model loader へ接続するか、`input_attribution/adapters/port_template.py` をコピーして小さな adapter を実装します。学習時の設定解決、環境生成、model input 前処理、Action decode、未加工テレメトリ取得は adapter に閉じ込めます。既存 `train.py`、`evaluate.py`、MetaDrive本体、SB3 の private helper は変更しません。

adapter の最小契約は次の通りです。

* `load_policy(config)`：環境を起動せず `ExistingPPO.load(config.model_path, env=None)` で保存済み PPO を読み、`CategoricalPolicyAdapter` で包む。返す policy adapter は、確率を返す `probabilities(observations)`、Action を返す `predict(observation, deterministic=True)`、再現性用の `fingerprint()`、評価モードを固定する `set_eval()` の4つを実装する。
* `make_env(config)`/`create_env(config)`：学習時と同じ観測生成・終了条件の環境を作る。`reset_env(env, seed)` は要求した seed を環境へ渡し、raw reset observation と info を返す。info または環境の readback から実際に適用された `actual_scenario_seed` を取得し、要求値 `requested_scenario_seed` と併せて保存する。取得できない場合は一致扱いにせず未検証にする。
* `preprocess_observation(observation, info)` または `to_model_input(...)`：モデルへ実際に渡す1次元 vector と前処理情報を返す。
* `telemetry(env, info, phase, step)`：加工前の target-lane 横ずれ、位置、向き、速度、進行度、逸脱/衝突/到達を返す。reset直後の初期 snapshot には少なくとも `position_xy_m`（または同じ意味の位置）、`target_lane_ordinal`、`target_lane_valid` を含め、可能なら `target_lane_offset_m`、`normalized_target_lane_error`、`road_segment_id`、`target_lane_heading_rad` も保存する。取れない値は欠測のままにする。
* `decode_action(action)`：Action番号と操舵・加減速の対応を返す。
* `close_env(env)`：各 closed-loop run の終了時に資源を解放する。
* `assert_contract(...)`：flat Box/Discrete、入力次元、dtype、policy の Action 数を確認する。`verify_schema_contract(schema, ...)` はまず `schema.validate_for_execution()` を呼び、その後 adapter が所有する観測生成ソースとの意味・順序・normalization 検証を行う。後者を実装できない移植先は `NotImplementedError` を送出して停止し、次元一致だけで通過させない。`source_paths()` は確認に使った既存ソースを manifest へ記録する。

移植先ごとの実装例は次のように、確認できない値を空欄/例外のまま保持します。source 未確認の index、意味、無効値を埋めるまで check を成功させません。

```python
import numpy as np

from input_attribution.policy import CategoricalPolicyAdapter
from input_attribution.adapters.port_template import PortAdapterTemplate

class MyAdapter(PortAdapterTemplate):
    def load_policy(self, config=None):
        model = ExistingPPO.load(config.model_path, env=None)
        policy = CategoricalPolicyAdapter(model)
        policy.set_eval()
        return policy

    def preprocess_observation(self, observation, info=None):
        vector, metadata = existing_observation_builder(observation, info)
        vector = np.asarray(vector, dtype=np.float32)
        expected_dimension = int(self.analysis_config.expected_dimension)
        if vector.ndim != 1 or vector.shape[-1] != expected_dimension:
            raise ValueError(
                f"model input must be a 1D vector of {expected_dimension}, got {vector.shape}"
            )
        return vector, metadata

    def verify_schema_contract(self, schema, *, expected_dimension=None):
        schema.validate_for_execution()
        expected = schema.dimension if expected_dimension is None else int(expected_dimension)
        if schema.dimension != expected:
            raise ValueError(
                f"schema dimension {schema.dimension} != expected model dimension {expected}"
            )
        self.verify_source_semantics(schema)
        return {"status": "verified", "dimension": schema.dimension}

    def verify_source_semantics(self, schema):
        raise NotImplementedError(
            "port-owned verification must compare every InputSpec with the "
            "existing observation producer before execution"
        )
```

262 次元では「3入力だから末尾」と割り当てません。既存観測生成ソースで3入力の正確な index、符号、normalization、clip、無効時の値、valid flag と同時に変える index、道路区間をまたぐ更新規則を確認し、`InputSpec` に source とともに記録します。非独立な flag/value の一部だけを変える pattern は schema が拒否します。未確認の値を0、0.5、全LiDAR no-hitで埋めることは禁止です。

## 推奨手順

1. 移植先の既存 Python、SB3、PyTorch、MetaDrive の版と import 元を記録します。既存環境を更新しません。
2. adapter と schema だけを接続し、`check --probe` で reset 後の少数 step を動かします。model input shape/dtype、前処理、Action、テレメトリキー、不足フィールドを JSON で確認します。
3. 259 では schema 全 index の一意被覆と model input 次元を一致させます。262 では unresolved placeholder が0件になるまで停止します。
4. P00 の少数 stepを実行し、要求 seed と環境から読み戻した `actual_scenario_seed`、開始車両位置、target lane の ordinal/valid/offset、確率、Action decode、終了条件を通常走行と照合します。初期 geometry または実seedが欠ける場合は P00 pairing を未検証として保存します。P00の環境 step は加工なしの走行だけを通します。
5. ①-A で保存観測から結果を再計算し、env.step()を呼ばず、元配列と model weights/normalization が不変であることを確認します。
6. ①-B は P00、追加特徴、重点 group を逐次実行します。runごとに reset/closeし、terminated/truncated 後に step せず、加工済み入力から物理評価を作りません。
7. `report` で日本語 HTML/MD、UTF-8 BOM CSV、SVG、media links、manifest/status を確認します。未実行・欠測が0へ置換されていないことを確認します。

## よくあるエラー

| 表示 | 主な原因 | 修正箇所 |
| --- | --- | --- |
| observation dimension mismatch | 学習時観測・schema・model input が不一致 | adapter の前処理、解析 TOML、schema |
| unresolved custom input | 262 template の意味が未確認 | 移植先観測生成ソースを調べ `InputSpec` |
| coupled indices / flag mismatch | 有効 flag と関連値を別々に変更 | patterns の対象 group |
| unsupported preprocessing / double normalization | VecNormalize や外部統計が不明 | adapter の前処理契約、学習時統計 |
| target-lane telemetry unavailable | 物理 target lane を取得できない | adapter telemetry。最近傍 current lane で埋めない |
| action distribution mismatch | deterministic predict と解析分布が不一致 | policy adapter の distribution 接続 |
| report shows N/A | 結果未実行、値欠測、媒体依存不足 | status.json、該当 stage、telemetry |

分からない入力の意味を数値や次元から推測せず、check の不足設定へ反映してから実験します。
