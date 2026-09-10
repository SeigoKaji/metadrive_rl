# 移植クイックスタート

移植先では、まず追加フォルダをコピーし、次の4接続だけを確認します。既存の環境生成、方策、学習、報酬本体を作り直しません。

```text
input_attribution/
├── adapter.py       # 環境・前処理・model入口の接続
├── collection.py    # baseline／①-Bのstep記録
├── policy_comparison.py # 保存x_tから①-AとJS
├── reward_adapter.py    # returned rewardとの内訳照合
├── schema.py        # Dと各indexの正本
├── config.py        # 小さなTOML契約
├── visuals.py / reporting.py / storage.py
├── configs/
└── docs/
```

読む順序はこの文書、`InputAttributionAdapter.prepare`／`predict`／`make_env`／`snapshot`／`frame`、移植先の実環境factoryと通常前処理、`extract_reward_terms`、`schema.py`、`test_core.py`／`test_schema.py`／`test_reporting.py`／`test_acceptance.py`です。必要なときだけ移植先の観測生成・報酬計算の該当関数を開きます。全リポジトリや長い過去ログを先に読みません。

接続するcallableの形は次のとおりです。

```python
adapter.prepare(raw: object) -> np.ndarray
adapter.predict(value: object, *, deterministic: bool | None = None) -> tuple[np.ndarray, int | np.ndarray]
adapter.make_env() -> object
adapter.snapshot(env: object) -> dict[str, object]
adapter.frame(env: object) -> np.ndarray
collect_baseline(adapter: object, **kwargs: object) -> dict[str, object]
collect_closed_loop(adapter: object, pattern, **kwargs: object) -> dict[str, object]
extract_reward_terms(info, returned_reward: float, *, terminated: bool, truncated: bool) -> dict[str, float] | None
```

`prepare`／`predict`はmodel入力境界、`make_env`はraw環境生成、`snapshot`／`frame`はstep後の読み取り、`collect_*`は共通走行、`extract_reward_terms`は返却rewardの内訳接続を担当します。

## 1. 実入力境界を固定する

`InputAttributionAdapter` は1 agent、1次元Box、SB3 MLP、Discrete action、identity preprocessingを対象にします。移植先のmodel observation spaceと実env observation spaceを読み、D、dtype、shape、action countを一致確認します。通常観測を作ってから既存wrapperの前処理を1回だけ通し、MLP直前の読み取りprobeで次を確認します。

- 対象indexへ指定した-1がそのまま届く
- 対象外は同じstepの通常値のまま
- adapter無介入の確率と通常 `model.predict(..., deterministic=True)` が一致する
- model／env／schemaが同じDで、padding・切り捨て・二重正規化がない

前処理がidentityでない、画像／MultiDiscrete／recurrent policyなどの場合は、経路を黙って迂回せず `UnsupportedAdapterError` と未対応理由を残します。読み取りprobeは解除し、重み・報酬・環境状態を書き換えません。

## 2. schemaとpatternを作る

`D=262` だけでは259の順序を移せません。`template_262_schema()` は全262行unknownです。移植先で各indexの位置／意味／正規化／参照レーン／valid flagを確認し、実コードとsource commit／file hashを記録した明示schemaを渡します。観測をpaddingまたはtruncateして合わせません。

公式259を本当に同一と確認できる場合だけ `standard_259_schema()` と `input_schema_259.toml` を使います。patternは次の4キーだけのplain mappingで渡せます。

```python
{
    "id": "HOST_P01",
    "name": "verified_group_name",
    "indices": [0, 1],
    "fixed_value": -1.0,
}
```

①-Aと①-Bに同じpattern listを渡し、元入力の独立copyへ毎step適用します。関連しそうな別indexを自動追加せず、対象値をclipしません。

## 3. reward providerを接続する

stepから返ったrewardを正解値として保存し、環境状態を更新せずに `extract_reward_terms(info, returned_reward, terminated=..., truncated=...)` へ当該stepのinfo snapshotを渡します。providerは係数・符号・終了時上書き順を反映した最終返却値の成分だけを返します。`step_reward` のような名前だけで推測せず、reward関数を再呼出しせず、差額を「その他」に足さず、欠測を0にしません。`validate_reward_terms` が `verified`、`unavailable`、`mismatch`、`nonfinite` を分けます。

## 4. 最小確認

まず人工schemaとSyntheticEnvで次を確認します。

```bash
python3 -m pytest input_attribution/tests/test_schema.py input_attribution/tests/test_core.py input_attribution/tests/test_adapter.py input_attribution/tests/test_reporting.py input_attribution/tests/test_acceptance.py -q
python3 -m input_attribution check --config input_attribution/configs/synthetic_demo.toml
python3 -m input_attribution run --config input_attribution/configs/synthetic_demo.toml
python3 -m input_attribution report --run-dir outputs/input_attribution/demo/synthetic/run-<timestamp>-<pid>
```

T13の高位index／別provider接続は `test_synthetic_262_high_index_and_alternate_provider_connect_a_b_js_report` で確認します。その後、official smokeをbaseline＋P01/P02で実行します。確認項目は-1到達、入力shape／action count、報酬合計、GIF、reward PNG、JS PNG、HTMLです。実環境の262確認済みとは報告せず、未接続の報酬内訳はunavailableと表示します。変更した実ファイル、実行結果、未解決事項だけを短く記録してください。
