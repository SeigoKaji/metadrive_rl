# 入力固定値介入の実験ガイド

この追加パッケージは、固定したMetaDrive＋SB3 PPO方策に対し、モデル直前のD次元入力の指定groupを毎step `-1.0` に置き換えたときの変化を記録します。`-1.0` は範囲外の故障／stress条件です。公式の欠測値だとか、方策が必ず欠測と解釈すると説明しません。`-100.0` など別の有限値も設定できますが、既定の初回実験は-1です。

baselineは入力を変更しない通常走行です。同じmodel、environment、scenario seed、policy seed、horizon、deterministic設定で1回だけ保存し、その観測列を①-Aと比較元に共有します。baselineの観測や行動をpattern実行へ再生せず、各走行は同じ初期条件から独立に始めます。

①-Aは保存済みbaselineの各stepの方策入力をコピーし、対象groupだけを固定値にして、全actionの確率 `q_t` を求めます。環境のreset／stepは呼びません。baseline確率 `p_t` と同時刻の `q_t` から自然対数のJS divergenceを計算し、`argmax(p_t) != argmax(q_t)` の同時刻へ印を付けます。これは判断分布の変化であり、①-Bの走行報酬を①-Aへ混ぜません。

①-Bは各patternで環境を最初から生成し、現在の観測を通常の前処理へ通してから対象groupだけを固定値にし、その入力から選んだactionを実際にstepへ渡します。次の観測はその走行から受けます。自然終了が50stepなら50/50だけを記録し、baselineの127stepに合わせた0埋めはしません。returned reward、terminated／truncated、車両snapshot、GIF frameをstep成功直後に保存します。

初期10groupの順序は全実験で共通です。

1. `road_edges`（0,1）
2. `lane_heading`（2）
3. `speed`（3）
4. `controls_history_yaw`（4--7）
5. `lane_lateral`（8）
6. `navigation`（9--18）
7. `lidar_front`（19--48,229--258）
8. `lidar_left`（169--228）
9. `lidar_rear`（109--168）
10. `lidar_right`（49--108）

各groupの意味と実装式は [INPUT_SCHEMA.md](INPUT_SCHEMA.md) を参照します。特にLiDAR前方は配列をまたぎます。標準259を名前の推測や連続sliceで再現しません。

主表はbaselineと各patternの6列程度の比較に留め、主図はbaseline／変更走行のreward PNG、①-AのpatternごとのJS PNG、GIF、縦スクロールHTMLです。summaryの累積報酬は実際にstepが返した値の合計で、内訳が取得できない場合は未接続と表示します。内訳がある場合も、returned rewardとの一致をstepごとに検証します。JSが未比較のstepや自然終了後のstep、報酬内訳の欠測を0へ補完しません。

画像や表を読むときは、①-AのJSと①-Bのrewardを同じ因果量として扱わないでください。1 scenario／episodeだけから一般化や統計的有意性を主張せず、観測groupの次元数差を独自スコアで補正しません。詳細な259行schemaは主表ではなく、HTMLやCSVから必要時に展開します。

実験前には `official_smoke.toml` の2groupで接続を確認し、実モデルが使用可能なら `official_fixed_stress.toml` のN=10へ進みます。モデルなしの確認には、MetaDriveの意味を付与しない人工schemaを使う `synthetic_demo.toml` を選び、baseline 127 step／affected走行50 stepの終了境界を確認します。出力先 `outputs/input_attribution/demo/synthetic` を公式実験と分離します。262移植は `host_262_migration_template.toml` をそのまま実行済み扱いにせず、移植先の実観測・model・前処理・明示schemaを確認してから行います。合成262確認には `synthetic_262_demo.toml` を使えますが、これは人工契約のテストです。
