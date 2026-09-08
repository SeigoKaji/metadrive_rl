# 入力依存度分析の設計

## 目的と対象

学習済み離散行動 PPO が、行動を決める直前に受け取ったベクトルのどの入力へ依存しているかを、同じ観測の比較と走り直しで確認します。モデルの重み、学習時の報酬、道路、終了条件は解析のために変更しません。第一対象は公式設定相当の map=`C`、scenario seed=5、交通量0、horizon=500、単一の離散9 Action ですが、指定モデルの学習時観測・前処理・Action 定義を優先します。

標準 schema は、実装ソースで確認した 259 次元です。公式接続の fixture では 0–8 が自車状態、9–18 が navigation、19–258 が LiDAR です。これはこの schema の記録であり、次元数だけから別環境の意味や並びを推測しません。特に index 2 の向き整合度と index 8 の横位置は、同じレーン参照だと仮定しません。各入力の source、単位、正規化、範囲、置換可否は `input_schema.json`/`input_schema.csv` に保存します。

259次元の新しい標準設定は `input_attribution/configs/input_attribution_official_259_variants.toml` です。旧初期参照を固定して再現する設定は `input_attribution/configs/input_attribution_official_259_legacy_freeze.toml` として分離し、同じ実験条件として混ぜません。新しい local neutral/reflection/fixed variant はエピソード全体の適用範囲を明示した別patternとして扱います。`index 2` は生角度ではなく確認済みの向き成分、`index 8` は現行schemaの正規化値であり、別PCの262次元へ数値を移植しません。

移植先の 262 次元は、開始時の目標レーンからの横ずれ、目標レーンへの向き誤差、有効フラグという3入力を含むテンプレートです。テンプレートは非末尾 index も表現できますが、index、符号、正規化、中立値、無効時の組、道路区間更新規則は未確認のまま成功扱いにしません。`input_attribution/schemas/custom_262_template.json` に残る `UNRESOLVED_` を adapter と schema で解決してから実験します。

## ①-A、①-B、③の違い

通常走行で、行動決定前のモデル入力 `x_t`、確率 `p_t`、Action、step 後のテレメトリを保存します。`t=0` は reset 直後、Action の適用区間は `[t,t+1]` です。加工した観測はコピーして保持し、元観測を in-place 変更しません。

①-A（offline）は各保存時刻について `x_t` と対象だけを置換した `x'_t` を同じ方策へ入力します。`env.step()` は呼ばず、次時刻は元の通常走行観測を使います。主表示は次の三つです。

* 元と置換後の argmax Action の一致/不一致と行動変更割合。分母は、その pattern で置換が適用された時刻（`applied`）とし、適用不能な時刻は `skipped` として分けます。
* 元観測の Action `a_t` について、`q_t[a_t]-p_t[a_t]` を符号付き百分率ポイント（pp）で保存します。正は置換後に元 Action の確率が増えた向きです。主表は実際に入力が変わった時刻の絶対値平均 `|q_t[a_t]-p_t[a_t]|` と、その母数（`meaningful`/`changed_exact` 件数）を表示し、符号付き平均は詳細表で確認します。
* `JS(p,q)=0.5 KL(p||m)+0.5 KL(q||m)`、`m=(p+q)/2` を自然対数で計算します。0 の項は0とし、値の範囲は `[0, ln(2)]` nats です。

pattern の `scope` と `on_inapplicable` は集計の母数と一緒に保存します。`scope=full_episode` は全保存時刻を対象にし、前提不成立なら `on_inapplicable=abort_pattern` でその pattern を中断します。`scope=explicitly_conditional` は宣言した道路・レーン等の条件が成立する時刻だけを対象にし、`continue_unmodified_with_warning` なら不成立時刻を元入力のまま通過させて `skipped` と理由を記録します。条件不成立を0件の影響として集計しません。

各 pattern の A 集計は、`target`（計画対象）、`eligible`（前提を満たし比較可能）、`applied`（置換を実行）、`changed_exact`（保存精度で実入力が変化）、`noop`（`applied - changed_exact`）、`skipped`（前提不成立または範囲外）を別々に示します。`meaningful` は設定した報告許容幅を超えた変化だけで、`changed_exact` を上書きしません。したがって、許容幅により0になった行と、実入力が本当に同じだった no-op を区別できます。

①-B（closed-loop）はパターンごとに環境を逐次 reset し、毎 step、その走行自身から得た最新観測のコピーへ同じ置換を適用して Action を実環境へ渡します。通常走行の未来観測を再生する比較ではありません。P00（変更なし）を対応する対照として保存し、分岐後の同じ step を同一状態の比較とは呼びません。評価は加工観測からではなく、adapter が返す未加工の物理テレメトリから計算します。

B で走行する pattern の集合は `closed-loop --patterns` または `closed_loop.patterns` で指定し、動画を保存する pattern の集合は `[video].patterns` で指定します。`video.enabled=true` のとき、`video.patterns` を省略すると走行した全 pattern、配列ならその ID だけが動画対象になります。動画を全て無効にする場合は `video.enabled=false` とし、動画選択を介入対象の選択と混同しません。

①-B の主指標は、目標レーン横ずれ RMS (m)、最大絶対横ずれ (m)、目標レーン逸脱回数/時刻、到達、ルート進行度、走行時間、平均速度・停止、終了理由です。道路外逸脱と衝突は目標レーン逸脱とは分けます。Action 切替や操舵指令差分は補助指標であり、横加速度やジャークとは呼びません。目標レーン参照が欠測の区間は最近傍レーンで埋めず、valid count/valid time と N/A を表示します。

### ①-B 指標の定義と既定値

`closed_loop` 設定の `departure_tolerance_ratio=0.05`、`departure_consecutive_steps=1`、`low_speed_m_s=0.5` を記録します。各 post-step の未加工 target-lane offset を `e_t` とすると、横ずれ RMS は `sqrt(sum(e_t^2) / valid_poststep_count)`、最大値は `max(abs(e_t))` です。valid な post-step がない場合は両方とも N/A です。逸脱は `abs(e_t) > lane_width_m / 2 * (1 + departure_tolerance_ratio)` を判定し、連続 `departure_consecutive_steps` step 以上の qualifying run を1イベントとして、そのイベント数と qualifying interval の時間を保存します。`first_departure_time_s` は最初のイベントの post 時刻、`departure_time_s` は全 qualifying step の `dt` 合計で、同じ列として扱いません。

各 step の `dt` は `post_time - pre_time`、valid time は valid な lane reference に対応する `dt` の合計、valid rate は `valid_time / total_duration` です。lane reference が一つも確認できない場合、valid time/rate を0へ置換せず N/A とします。低速時間は速度 `< low_speed_m_s` の post-step `dt` の合計です。操舵の変化量は正規化済み command unit の `sum(abs(u_t - u_{t-1}))` であり、横加速度・ジャークではありません。到達は target lane の到達条件と lane identity が確認できた場合だけ成功とし、wrong-lane arrival は別フラグで保存します。

参照観測を使う pattern の `compatibility_keys` 既定値は `road_segment_id` と `target_lane_ordinal` です。これらの context が一致しない道路区間では reference pattern を skip し、確率差/JS/closed-loop 差を0にしません。初期 seed、初期 geometry、target lane の一致が確認できない P00/paired 比較は `initial_comparison_verified=false` として N/A の理由を表示します。

③（Integrated Gradients）は任意の補足です。対象 episode/step と保存された baseline を明示し、元観測の argmax Action `a*` を固定して `F(x;a*) = z[a*] - logsumexp(z[other actions])` を追跡します。baseline 未指定時にゼロベクトルへ黙ってフォールバックしません。bool・カテゴリ・有効フラグは同じ値の baseline とし、直線補間が実在状態とは限らないことを記録します。`sum(IG)` と `F(x)-F(baseline)`、completeness 残差、積分点数を保存し、IGを①-Aの順位や①-Bの性能へ合算しません。Captum は遅延 import の任意依存です。

保存済み通常観測の再利用は、report表示の再生成、①-Aの新variant、①-Bの新走行を別操作にします。`report --run-dir OLD --output-dir DIR` は保存値からDIRだけへ出力し、`offline --run-dir OLD --config NEW` はモデル・入力意味/順序/正規化・結合条件・前処理が一致するときだけ参照観測をコピーした子runを作ります。variant追加は許可されますが、意味定義の変更は拒否します。①-Bは子runの設定で環境を再走行し、旧Bの数値を新条件へ流用しません。

## 置換パターンと集計

最初に P00 を指定します。追加特徴はスキーマの意味確認済み置換値だけを使い、非独立の有効フラグは関連値と組で変更します。保存参照を使う場合は episode/step/reference id を残し、同じ一つの実観測から対象群を取ります。全次元一律0、未知の中立値、任意角度への LiDAR 割当ては行いません。実際に変更した index、適用件数、実変更件数、no-op件数、skip reason を記録します。

LiDAR は単一 index、方向 sector/group、全240次元を同じ影響として順位付けしません。`information_removal`（確認済み no-detection 値）と `diagnostic_virtual_detection`（仮想検出値、物体追加を意味しない）を `variant_classification` として分け、対象 sector と実変更件数を保存します。これは LiDAR の物理的な物体検出結果や道路上の障害物追加とは解釈しません。

主影響欄では、適用されたが元値と同じ no-op、または `meaningful` 判定に届かない変化を影響0とは表示せず N/A（評価対象外）とします。一方、詳細表・JSON・CSV には生の実測値として `changed_exact=0`、`noop`、符号付き pp、JSを保存します。これにより「変化が無かった」という観測と「比較できなかった」という状態を後から区別できます。

全時刻集計と実変更時刻だけの集計を分け、episode 平均と step 加重平均を別列にします。入力1個の結果と大きなグループの結果は同じ順位へ混ぜません。未実行、範囲外、依存不足、テレメトリ欠測は0ではなく理由付き N/A です。数値から言えるのは「このモデル・対象場面・指定置換条件で影響が確認された」までです。

①-B は「到達」と「到達後のレーン維持」を別に判定します。到達は確認済みの target-lane identity と到達条件を満たしたか、維持は到達後または走行区間の valid telemetry で横ずれ RMS・逸脱回数・逸脱時間を評価したかで示します。未到達、wrong-lane arrival、参照欠測、衝突・道路外逸脱・停止・中断は成功や維持の0へ潰さず、それぞれの状態と valid 分母を保存します。

## 参考手法との関係

Greydanus et al., *Visualizing and Understanding Atari Agents* (ICML 2018) の入力摂動で方策・価値の変化を見る着想を参照しました。原論文の画像局所 Gaussian blur、pre-softmax logit 二乗距離、critic 分析は今回の観測ベクトル置換、確率 JS、目標レーン指標とは別の設計です。JSを原論文固有の指標とは書きません。

Sundararajan et al., *Axiomatic Attribution for Deep Networks* (ICML 2017) の straight-path Integrated Gradients と completeness を③の理論基礎にします。PPOの選択 Action の対数オッズ、保存観測を baseline とする選定、フラグ固定は今回の適用設計です。

Atrey et al., *Exploratory Not Explanatory* は、saliency から得た仮説を counterfactual 実験で確かめる必要性の参考です。今回の sensor vector のコピー置換と closed-loop 実験は、その論文の手順そのものではありません。Captum API の `n_steps`、`method`、`gausslegendre`、`internal_batch_size`、`return_convergence_delta` を任意 IG 接続の契約として扱います。

参考 URL: [Greydanus (PMLR)](https://proceedings.mlr.press/v80/greydanus18a.html)、[Sundararajan (PMLR)](https://proceedings.mlr.press/v70/sundararajan17a.html)、[Atrey (arXiv)](https://arxiv.org/abs/1912.05743)、[Captum Integrated Gradients API](https://captum.ai/api/integrated_gradients.html)。この文書は該当節と API を確認した範囲の記録であり、論文全体を読了したという主張はしません。
