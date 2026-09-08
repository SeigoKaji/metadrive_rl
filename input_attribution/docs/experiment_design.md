# 入力依存度分析の設計

## 目的と対象

学習済み離散行動 PPO が、行動を決める直前に受け取ったベクトルのどの入力へ依存しているかを、同じ観測の比較と走り直しで確認します。モデルの重み、学習時の報酬、道路、終了条件は解析のために変更しません。第一対象は公式設定相当の map=`C`、scenario seed=5、交通量0、horizon=500、単一の離散9 Action ですが、指定モデルの学習時観測・前処理・Action 定義を優先します。

標準 schema は、実装ソースで確認した 259 次元です。公式接続の fixture では 0–8 が自車状態、9–18 が navigation、19–258 が LiDAR です。これはこの schema の記録であり、次元数だけから別環境の意味や並びを推測しません。特に index 2 の向き整合度と index 8 の横位置は、同じレーン参照だと仮定しません。各入力の source、単位、正規化、範囲、置換可否は `input_schema.json`/`input_schema.csv` に保存します。

移植先の 262 次元は、開始時の目標レーンからの横ずれ、目標レーンへの向き誤差、有効フラグという3入力を含むテンプレートです。テンプレートは非末尾 index も表現できますが、index、符号、正規化、中立値、無効時の組、道路区間更新規則は未確認のまま成功扱いにしません。`custom_262_schema_template` に残る `UNRESOLVED_` を adapter と schema で解決してから実験します。

## ①-A、①-B、③の違い

通常走行で、行動決定前のモデル入力 `x_t`、確率 `p_t`、Action、step 後のテレメトリを保存します。`t=0` は reset 直後、Action の適用区間は `[t,t+1]` です。加工した観測はコピーして保持し、元観測を in-place 変更しません。

①-A（offline）は各保存時刻について `x_t` と対象だけを置換した `x'_t` を同じ方策へ入力します。`env.step()` は呼ばず、次時刻は元の通常走行観測を使います。主表示は次の三つです。

* 元と置換後の argmax Action の一致/不一致と行動変更割合。
* 元観測の Action `a_t` について、`q_t[a_t]-p_t[a_t]` を百分率ポイント（pp）で表示します。正は置換後に元 Action の確率が増えた向きです。
* `JS(p,q)=0.5 KL(p||m)+0.5 KL(q||m)`、`m=(p+q)/2` を自然対数で計算します。0 の項は0とし、値の範囲は `[0, ln(2)]` nats です。

①-B（closed-loop）はパターンごとに環境を逐次 reset し、毎 step、その走行自身から得た最新観測のコピーへ同じ置換を適用して Action を実環境へ渡します。通常走行の未来観測を再生する比較ではありません。P00（変更なし）を対応する対照として保存し、分岐後の同じ step を同一状態の比較とは呼びません。評価は加工観測からではなく、adapter が返す未加工の物理テレメトリから計算します。

①-B の主指標は、目標レーン横ずれ RMS (m)、最大絶対横ずれ (m)、目標レーン逸脱回数/時刻、到達、ルート進行度、走行時間、平均速度・停止、終了理由です。道路外逸脱と衝突は目標レーン逸脱とは分けます。Action 切替や操舵指令差分は補助指標であり、横加速度やジャークとは呼びません。目標レーン参照が欠測の区間は最近傍レーンで埋めず、valid count/valid time と N/A を表示します。

### ①-B 指標の定義と既定値

`closed_loop` 設定の `departure_tolerance_ratio=0.05`、`departure_consecutive_steps=1`、`low_speed_m_s=0.5` を記録します。各 post-step の未加工 target-lane offset を `e_t` とすると、横ずれ RMS は `sqrt(sum(e_t^2) / valid_poststep_count)`、最大値は `max(abs(e_t))` です。valid な post-step がない場合は両方とも N/A です。逸脱は `abs(e_t) > lane_width_m / 2 * (1 + departure_tolerance_ratio)` を判定し、連続 `departure_consecutive_steps` step 以上の qualifying run を1イベントとして、そのイベント数と qualifying interval の時間を保存します。`first_departure_time_s` は最初のイベントの post 時刻、`departure_time_s` は全 qualifying step の `dt` 合計で、同じ列として扱いません。

各 step の `dt` は `post_time - pre_time`、valid time は valid な lane reference に対応する `dt` の合計、valid rate は `valid_time / total_duration` です。lane reference が一つも確認できない場合、valid time/rate を0へ置換せず N/A とします。低速時間は速度 `< low_speed_m_s` の post-step `dt` の合計です。操舵の変化量は正規化済み command unit の `sum(abs(u_t - u_{t-1}))` であり、横加速度・ジャークではありません。到達は target lane の到達条件と lane identity が確認できた場合だけ成功とし、wrong-lane arrival は別フラグで保存します。

参照観測を使う pattern の `compatibility_keys` 既定値は `road_segment_id` と `target_lane_ordinal` です。これらの context が一致しない道路区間では reference pattern を skip し、確率差/JS/closed-loop 差を0にしません。初期 seed、初期 geometry、target lane の一致が確認できない P00/paired 比較は `initial_comparison_verified=false` として N/A の理由を表示します。

③（Integrated Gradients）は任意の補足です。対象 episode/step と保存された baseline を明示し、元観測の argmax Action `a*` を固定して `F(x;a*) = z[a*] - logsumexp(z[other actions])` を追跡します。baseline 未指定時にゼロベクトルへ黙ってフォールバックしません。bool・カテゴリ・有効フラグは同じ値の baseline とし、直線補間が実在状態とは限らないことを記録します。`sum(IG)` と `F(x)-F(baseline)`、completeness 残差、積分点数を保存し、IGを①-Aの順位や①-Bの性能へ合算しません。Captum は遅延 import の任意依存です。

## 置換パターンと集計

最初に P00 を指定します。追加特徴はスキーマの意味確認済み置換値だけを使い、非独立の有効フラグは関連値と組で変更します。保存参照を使う場合は episode/step/reference id を残し、同じ一つの実観測から対象群を取ります。全次元一律0、未知の中立値、任意角度への LiDAR 割当ては行いません。実際に変更した index、適用件数、実変更件数、no-op件数、skip reason を記録します。元から置換値と同じ値だった no-op は「この条件では評価不能」と記載し、影響0や不要の証拠にしません。

全時刻集計と実変更時刻だけの集計を分け、episode 平均と step 加重平均を別列にします。入力1個の結果と大きなグループの結果は同じ順位へ混ぜません。no-op、未実行、範囲外、依存不足、テレメトリ欠測は0ではなく理由付き N/A です。数値から言えるのは「このモデル・対象場面・指定置換条件で影響が確認された」までです。

## 参考手法との関係

Greydanus et al., *Visualizing and Understanding Atari Agents* (ICML 2018) の入力摂動で方策・価値の変化を見る着想を参照しました。原論文の画像局所 Gaussian blur、pre-softmax logit 二乗距離、critic 分析は今回の観測ベクトル置換、確率 JS、目標レーン指標とは別の設計です。JSを原論文固有の指標とは書きません。

Sundararajan et al., *Axiomatic Attribution for Deep Networks* (ICML 2017) の straight-path Integrated Gradients と completeness を③の理論基礎にします。PPOの選択 Action の対数オッズ、保存観測を baseline とする選定、フラグ固定は今回の適用設計です。

Atrey et al., *Exploratory Not Explanatory* は、saliency から得た仮説を counterfactual 実験で確かめる必要性の参考です。今回の sensor vector のコピー置換と closed-loop 実験は、その論文の手順そのものではありません。Captum API の `n_steps`、`method`、`gausslegendre`、`internal_batch_size`、`return_convergence_delta` を任意 IG 接続の契約として扱います。

参考 URL: [Greydanus (PMLR)](https://proceedings.mlr.press/v80/greydanus18a.html)、[Sundararajan (PMLR)](https://proceedings.mlr.press/v70/sundararajan17a.html)、[Atrey (arXiv)](https://arxiv.org/abs/1912.05743)、[Captum Integrated Gradients API](https://captum.ai/api/integrated_gradients.html)。この文書は該当節と API を確認した範囲の記録であり、論文全体を読了したという主張はしません。
