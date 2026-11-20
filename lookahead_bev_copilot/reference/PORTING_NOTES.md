# 評価BEVの移植ノート

[bev_overlay.patch](bev_overlay.patch) は移植元の実装差分です。下記のファイル名やフィールド名は移植元のものです。派生で名前や構造が異なる場合は、同じ役割を持つ既存箇所に必要な変更だけを入れます。

## 評価ループと描画データ

1. `evaluate.py::_evaluate` が `make_evaluation_env()` から環境を作ります。`LookaheadEnv.reset()` は MetaDrive の reset 後に `MetaDrivePreviewProvider.reset()` を呼び、先読み経路を構築します。
2. reset 後に `env.get_wrapper_attr("reference_path")` で同じ `RouteBuildResult.path` を読み取り、`evaluation_results.py::_EpisodeVisualizationRecorder` に渡します。収録や先読みを無効にした場合や、provider が経路を持たない場合は既存の扱いを使います。
3. `model.predict(obs)` → `env.step(action)` で返った `info` は `make_step_telemetry()` が保持します。`record_frame()` は行動後の BEV に、固定経路と行動決定時の点を `overlay_lookahead_bev()` で描き、`compose_telemetry_panel()` で右パネルを加えます。

### 注視点の時刻と無効値

| 実データ | 読み方 |
| --- | --- |
| `info["lookahead_learning"]["pre_step"]["preview"]` | 今表示している行動の選択に使った観測。`preview_valid` が `true` の場合に、`q_xy` の有限な世界座標 `[x, y]`（m）を描きます。 |
| `info["lookahead_learning"]["post_step"]["preview"]` | 行動後に返る観測。episode 継続中は次の行動の `pre_step` になります。 |
| `preview_valid` が `false` | 輪を省き、`Decision lookahead (unavailable)` と表示します。進行方向の判定などで無効になった点には、診断用の `q_xy` が残る場合があります。 |

reset の `pre_step` は `null` です。最初の action の `pre_step` には reset 時の観測が入ります。名前の異なる派生では、モデルへ渡した観測と同時に計算した世界座標を保存し、その行動のフレームに渡します。preview や `q_xy` が `null` または欠損なら輪を省きます。自動 reset 時は、終了ステップのフレームに終了した episode の状態を使います。

## 経路とカメラ

- `lookahead_learning/adapter.py::build_fixed_navigation_route()` は `Navigation.checkpoints` の隣接ノード対から有向道路を引き、開始車線と同じ番号の車線を選びます。端点・接線・車線数・車線幅を確認し、接続が確かめられた範囲を `ReferencePath` に保持します。MetaDrive 標準十字路では交差点内にも直進と左右折の車線があります。未検証境界を越える先読みは無効です。描画も同じ `ReferencePath.lanes` の範囲で止め、接続が未検証の区間に線を引きません。
- `evaluation_results.py::_configure_route_view()` は各車線の `lane.position(s, 0.0)` を両端を含めて 1 m 以下の間隔でサンプリングし、その点列の外接矩形の中心をカメラ位置にします。移植元は 600×600 px、片側 24 px の余白、上限 5 px/m で縮尺を決め、初回 `env.render(mode="topdown", ...)` に渡します。MetaDrive は地図全体の寸法も見て実縮尺を決めます。
- `evaluation_visualization.py::overlay_lookahead_bev()` は、MetaDrive 0.4.3 の北向き `TopDownRenderer` が使う描画面の `pos2pix(x, y)` で原点・実縮尺・Y 軸反転を反映します。Pygame の `blit` と同じ整数クロップ原点を引いて BEV 画素位置を得ます。経路線は背景と一致するピクセルに描き、車両と履歴の色を残します。有効な注視点は橙色の輪で重ね、右のパネルに凡例を載せます。
- 追従・回転・クロップが違う派生では、現在のフレームを作ったレンダラの変換に線と点を通します。描画前に世界座標を扱えるレンダラなら、経路は車両の下の地図レイヤー、点は注視点のレイヤーに置けます。合成後に描画する場合は、回転・リサイズ・クロップの順序も含めて同じ変換を使用します。MetaDrive の私有属性は、同じ描画経路だと確認できた場合に限って参照します。

## 出力との接続と確認

移植元では `outputs/<name>/evaluation/evaluation_steps.jsonl` に step ごとの `lookahead_learning` があり、注視点と有効フラグを照合できます。`evaluation.json` の `episodes[*].visualization` に出力情報があります。各 episode の保存先は `episodes/episode_<連番4桁>[_<scene_group>]_scenario_<seed6桁>/` です。

| 出力 | 合成フレームの渡し方 |
| --- | --- |
| `frames/frame_000001.png` 以降 | Pillow に RGB の `H×W×3` 配列を渡します。MetaDrive の `WorldSurface.to_cv2_image()` の戻り値も RGB です。 |
| `evaluation.gif` | `replace_latest_recorded_frame()` で MetaDrive の `_screen_frames` の末尾を RGB 合成フレームに差し替え、`generate_gif()` で保存します。 |
| `evaluation.mp4` | 既存の OpenCV `VideoWriter`（移植元は `mp4v`）に RGB→BGR 変換した合成フレームを渡します。 |

1行動の長さを `dt = physics_world_step_size × decision_repeat` 秒とすると、1行動につき1枚を記録する移植元では GIF の表示時間が `1000 × dt` ms、MP4 は `1 / dt` fps です。各 episode の `visualization` で、使用する形式ごとに `status` と `frame_count` を確認します。派生でフレームを間引く場合は既存の記録間隔を使います。保存名・コーデック・出力形式もその評価器に合わせてください。移植元の時間対応と出力を確認するテストは `tests/test_evaluation_visualization.py::test_lookahead_recording_uses_the_point_observed_for_each_action` です。
