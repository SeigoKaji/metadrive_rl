# BEV の短時間確認

既存の学習・評価出力を上書きしないよう、一時ディレクトリを出力先に使う。派生先の Python 環境で実行する。

## 1. 静止から発進する2ステップ

参考パッチを適用できる同構成では、プロジェクト直下で次のテストを実行する。既存ファイルと名前が異なる場合は、派生先の評価入口で同じ条件を確認する。

```sh
python -B -m pytest -q tests/test_evaluation_visualization.py::test_lookahead_recording_uses_the_point_observed_for_each_action tests/test_evaluation_visualization.py::test_one_step_connects_to_metadrive_043_recorded_frame tests/test_evaluation_visualization.py::test_disabled_episode_recorder_creates_no_visualization_artifacts
```

1. 静止状態から前進する行動を2回続け、各ステップの `lookahead_learning.pre_step.preview` と `post_step.preview`、対応するPNGを確認する。同梱テストは `SXS`、シナリオseed 5、`horizon = 2`、`lookahead_time_s = 5.0`。評価用のモデルは前進を選ぶように固定して作成し、学習はしない。
2. 1枚目は行動決定時の注視点が無効なので、経路線だけが見える。1ステップ目の行動後に有効になった点は、2ステップ目の行動決定時の点として2枚目に出る。`q_xy` が残っていても `preview_valid = false` の点は描かない。PNG、GIF、MP4はすべて2フレームになる。
3. `[lookahead]` がない既存設定では、BEVに経路線・注視点・凡例を加えず、元の画角とテレメトリを保つ。録画無効時は画像・動画の成果物を作らない。

## 2. 十字路で座標と描画を照合する

参考実装と同じ構成で、学習済みモデルを使わずに再現する場合は、プロジェクト直下で次を実行する。既存の評価レコーダーで左折部分を撮影し、各PNGの注視点マーカーの中心を行動決定時の実座標と照合する。PNG・GIF・MP4は実行ごとに新しい一時ディレクトリへ出力する。開始時にその場所を表示し、途中で失敗した場合も生成済みファイルを診断用に残す。派生先でモジュール名や録画の入口が異なる場合は、次の確認項目を現地の実装に当てはめる。スクリプトの導入は任意。

```sh
python -B -m lookahead_bev_copilot.reference.check_turn
```

1. 実環境を `SXS`、シナリオseed 0、固定距離 `lookahead_m = 6.0` でリセットする。左折の手前から交差点内までの短い区間を記録し、リセット時に選んだ固定経路の中心線と、そのステップの行動決定時の有効な注視点を重ねる。学習済みチェックポイントの代わりに、短い動作確認用の固定ルールで車両を動かしてもよい。
2. 同じカメラ変換で、車両の行動後の世界座標と `pre_step.preview.q_xy` を別々に画素へ投影する。この構成の MetaDrive では `WorldSurface.pos2pix` で実際の原点・縮尺・Y軸反転を適用し、BEVの切り出し左上を引く。右側のテレメトリパネルは含めない。投影した車両位置が描画された車両に重なり、投影した注視点がマーカーの中心に一致することをPNGで確かめる。
3. 進入直線から左折円弧へ経路線がつながり、前進に合わせて注視点が円弧へ移ることを確認する。経路が未確定の区間まで線を延ばさない。経路・有効な点を取得できない場合はその箇所を空欄にし、表示できなかった条件を記録する。

同梱見本の条件は MetaDrive 0.4.3（`85e5dadc`）、オフスクリーン描画、北を上にした固定カメラ。BEVは600×600 px、テレメトリを含む全体は920×600 px。カメラは固定経路全体が入るように合わせている。初期車速は10 m/s、スポーン車線の縦座標は35 m、ステアリングは81段階。デモでは Pure Pursuit の参照操舵に従う固定ルールで車両を動かした。GIFはステップ77〜104の28フレーム（10 Hz、2.8秒）。PNGはステップ97。青緑の線が固定参照経路、橙の輪が行動決定時の注視点、緑の矩形が行動後の車両。

[ステップ97のPNG](intersection_step_0097.png)・[左折のGIF](intersection_lookahead.gif)
