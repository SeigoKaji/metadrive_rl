# 観測スキーマ

入力寄与解析は、観測の次元数や意味を Python に埋め込みません。各 `.toml` は全 index を一度ずつ定義し、block（明示的な feature 名）と range（連続 feature）を展開します。通常の解析は `resolved = false` の placeholder を含む schema を必ず拒否します。

## `metadrive_default_259.toml` の適用条件

この schema はローカルの MetaDrive `0.4.3`、commit `85e5dadc` の `StateObservation` と `NodeNetworkNavigation` の実装を確認したものです。次の条件を**すべて**満たす公式相当の vector observation だけに使用してください。

- `image_observation = false`、`agent_observation = None`、`random_agent_model = false`
- navigation は `NodeNetworkNavigation`（10 次元）
- side detector と lane-line detector はともに `num_lasers = 0`。有効にする場合は detector distance も正であり、観測 shape 分岐を別途確認すること
- LiDAR は `num_lasers = 240`、distance は正、`num_others = 0`、`add_others_navi = false`
- MetaDrive の標準 append order を使うこと: state、navigation、（`num_others > 0` なら other-vehicle 情報）、cloud points

この条件では index 0--1 は left/right road boundary distance、2 は `[0, 1]` に正規化された heading difference、3--8 は speed / steering / action history / **符号なし** yaw rate / current-lane lateral position です。index 9--18 は checkpoint ごとに forward projection、right projection、curve radius、curve direction、curve angle の順です。straight road の curve direction / curve angle は実装で正規化後 `0.5` になります。index 19 以降は LiDAR です。

LiDAR の angle metadata は vehicle heading の `0°` から ray ごとに `+1.5°` です。vehicle local `+y` が right を向く MetaDrive の慣例では clockwise scan として記録しています。図は必ず schema の `angle_deg` を使うため、コード中に LiDAR の開始 index や ray 数はありません。

## 262 次元版を記入する方法

`custom_262_template.toml` は完成済みの 262 schema ではありません。3 入力の意味が確認できるまで `UNRESOLVED_*` と `resolved = false` を残し、`validate-schema` / `run` が失敗することが正しい動作です。

追加 3 入力が**末尾に追加**される実装なら、次の block だけを編集します。

```toml
[[blocks]]
name = "custom_lane_keeping"
start = 259
kind = "custom"
group = "custom_lane_keeping"
resolved = true
description = "Verified wording from the custom environment source."
feature_names = ["verified_feature_0", "verified_feature_1", "verified_feature_2"]
```

追加 3 入力が**LiDAR の直前に挿入**される実装なら、同じ custom block の `start = 19`、LiDAR range の `start = 22` に変更します。それ以外の Python source は変更しません。いずれの場合も、確認した正確な feature 名・意味・group・description を記入し、必要なら `constants` を schema-constant 閉ループ介入用に明示します。推測した仮名のまま解析しないでください。

汎用の未解決 template は次でも作れます。

```bash
.venv/bin/python analyze_input_attribution.py schema-template \
  --dim 262 --output observation_schemas/custom_262.toml
```

生成物は意図的に generic / unresolved です。そこから、上記の検証済み block/range だけを編集してください。
