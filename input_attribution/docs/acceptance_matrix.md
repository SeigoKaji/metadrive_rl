# 受入確認マトリクス

実行時点の全体テスト件数・fake/実機区分・任意 IG wheel の provenance は [validation_report.md](validation_report.md) に記録します。

| 区分 | 確認内容 | fake/保存結果で確認 | MetaDrive実機 | 成果物/理由 |
| --- | --- | --- | --- | --- |
| A 入力・移植 | 259 schema の全 index、262 template の非末尾3入力、範囲・前処理・coupled flag | fake schema fixture で確認。262 template は未解決を拒否 | 259は実モデル・観測・ソースを照合済み。実262は移植先確認が必要 | `check.json`、`input_schema.*`。実機262検証済みとは書かない |
| B ①-A | P00の差分0、確率差/JS、no-op、元配列不変、env.step未呼出 | fake policy と保存観測で確認 | 実259の127保存観測で確率一致、272パターン評価 | `01_offline/<analysis_id>/result.json` |
| C ①-B | reset/seed、加工後 Action、最新観測、未加工 target-lane telemetry、終了後step禁止 | fake adapter の一連の走行で確認可能 | 実259の6パターンを実走行。P00の全127step trace一致 | `02_closed_loop/<analysis_id>/.../trajectory.json`、`summary.json` |
| D ③ IG | Captumなしで主工程維持、線形 completeness、カテゴリ補間禁止 | fake differentiable policy と任意依存なしを確認 | 実259の指定2時刻でCaptum IG収束・重み不変 | `03_ig/<analysis_id>/result.json`。失敗しても①結果を保持 |
| E 成果物 | HTML/MD、UTF-8 BOM CSV、inline SVG、P00差、N/A、manifest/status、再生成ID、portable ZIP | fixture と pytest で確認 | 実259の7 GIF、全frame overlay、HTMLリンク欠落0を確認 | `report.*`、`reports/<report_id>/`、ZIP |

## 状態の読み方

`status.json` の stage が `success` で `analysis_id`/`relative_dir` を持つ場合、report はその最新成功ディレクトリだけを読みます。過去の analysis ID を同じ表へ混ぜません。`failed`、`pending`、`running`、`skipped` は未完了または未実行として表示し、古い成功結果へ黙って戻しません。値が取れない欄は N/A と理由を表示します。

fake 実行で得た数値は、実 MetaDrive 車両の性能結果として報告しません。実機結果がない段階では、check/fixture の接続契約・配列処理・レポート生成が確認済みであることだけを記載します。

保存済み synthetic_259 の report 回帰では、P00 の適用件数 4・no-op 4、互換コンテキスト不足の reference pattern は `未実行（skip）` と理由付き N/A、target-lane `valid_count=0` の closed-loop 横ずれ・P00差は N/A になりました。これは fake adapter の保存形式を確認する数値であり、実車両の性能推定ではありません。担当レポート回帰は `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest input_attribution/tests/test_reporting.py -q -p no:cacheprovider` で確認し、repo 全体の件数は runtime/core 統合後の `docs/validation_report.md` に記録します。
