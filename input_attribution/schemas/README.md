# 配布 schema

`official_259.json` は、導入済み MetaDrive 0.4.3 の source-verified 259 要素
mapping です。解析用 TOML の `schema.path` から読み込み、モデルの Box shape と
全 index を照合します。

`custom_262_template.json` は移植先の確認前に実行できないテンプレートです。
目標レーン横ずれ・向き誤差・有効フラグの意味 ID は記録していますが、index、
符号、normalization、neutral、invalid 値、coupled index は `UNSET` です。
次元から標準259要素へ挿入位置を推測しないでください。移植先の観測生成 source
を確認した後、全要素を埋めた新しい `InputSchema` JSON と TOML の `schema.path`
を作成してください。

`adapters.fixtures.custom_262_schema_template` の非末尾 index は、推測禁止を
テストするためだけの synthetic fixture です。実機の 262 schema mapping では
ありません。

`fake_262_resolved.json` は fake adapter の一連テスト用にだけ使う、非末尾
index の synthetic mapping です。実機262の確認済みmappingやモデルの意味を
表しません。
