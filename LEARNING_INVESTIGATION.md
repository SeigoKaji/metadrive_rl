# `model.learn()` 学習内容・環境仕様・ソース追跡調査

最終確認日: 2026-08-20（Asia/Tokyo）

## 1. この文書の目的

この文書は、このリポジトリの `model.learn()` が何を学習しているかを、プロジェクトコード、MetaDrive、Stable-Baselines3（SB3）の実ソースまで追跡して整理したものである。特に、次を明確にする。

- 学習で更新されるものと、更新されないもの
- Actor-Criticという用語と、この実装での具体的な意味
- PPOが収集するデータ、GAE、return、loss、optimizer更新
- MetaDriveのReward、cost、終了条件
- `official` の259次元Observationと `generalization` の261次元Observation
- LiDARの「正規化距離」と最大50 m設定
- `Discrete(9)` Actionとsteeringの左右
- 現在残っている学習・評価成果物と、そこから言える範囲
- 初めて読む人がソースを追う順番

数式は、VS Codeに組み込まれたMarkdown Math（既定で有効）がKaTeXで組版できるよう、ディスプレイ数式を `$$ ... $$` で記述している。

本書の状態ラベルは次の意味で使う。

| ラベル | 意味 |
| --- | --- |
| **confirmed** | 現在のローカルソース、設定、または現存artifactを直接確認した |
| **reported** | `RUN_REPORT.md` など、既存の実行記録に記載されている |
| **inferred** | confirmedな実装と数値から一意に導出できるが、専用runtime probeは行っていない |
| **unverified** | 現在のsnapshotや今回の静的調査では確認できない |

対象versionは次のとおりである。

| 項目 | 対象 |
| --- | --- |
| MetaDrive | sibling source `../metadrive`、commit `85e5dadc6c7436d324348f6e3d8f8e680c06b4db`、version 0.4.3 |
| Stable-Baselines3 | project内 `.venv` の2.9.0 |
| Python | 3.12.3 |
| 主な根拠 | [requirements.lock.txt](requirements.lock.txt)、[MetaDrive source](../metadrive/metadrive)、[SB3 source](.venv/lib/python3.12/site-packages/stable_baselines3) |

## 2. 結論

このプロジェクトの `model.learn()` は、MetaDriveの車両物理や道路生成規則を学ぶのではない。モデルフリー強化学習アルゴリズムのPPOを使い、次の2つのニューラルネットワークを学習する。

1. **Actor（Policy）**
   - Observationから9個のActionのlogitを出す。
   - logitからCategorical分布 `π(a|s)` を作る。
   - 学習中は分布からAction IDをsampleし、評価中は通常、最大確率のAction IDを選ぶ。

2. **Critic（Value function）**
   - Observationから状態価値 `V(s)` を1個出す。
   - `V(s)` は「現在の状態から将来得られる割引累積Rewardの期待値」を近似する。
   - Actionを直接決めず、advantage推定とActor更新を安定させる。

したがって、PPOが直接学習する中心は次の写像である。

~~~text
Actor:  Observation → 9 Actionの確率分布
Critic: Observation → 将来Rewardの期待値
~~~

`official` profileではObservationは259次元、`generalization` profileでは261次元である。Action spaceはどちらも `Discrete(9)` である。

最も重要な注意点は、MetaDriveが `info["cost"]` を計算していても、現在のSB3 PPOはそのcostをrollout bufferやlossに入れていないことである。現在のモデルが最適化するのは**Rewardのみ**であり、constrained RLやsafe RLではない。

## 3. 責任分界

| 層 | このプロジェクトでの責任 |
| --- | --- |
| 自作コード | profile選択、環境生成、SB3 PPO生成、`model.learn()` 呼び出し、保存、評価、artifact出力 |
| MetaDrive | Observation、Action変換、車両物理、道路・traffic生成、Reward、cost、terminated/truncated |
| SB3 | Actor/Critic、rollout収集、GAE、return、PPO loss、minibatch、backpropagation、optimizer step |
| PyTorch | tensor演算、自動微分、gradient、Adamによるparameter更新 |

自作の [train.py](train.py) は [`_run_training()`](train.py) 内で環境とPPOを作り、`train.py:289` で次を呼ぶ。

~~~python
model.learn(
    total_timesteps=args.timesteps,
    log_interval=args.log_interval,
)
~~~

PPO lossやbackpropagationは、このリポジトリ側では再実装していない。

## 4. 設定から `model.learn()` までの経路

### 4.1 全体のデータフロー

~~~text
configs/phase0_config.py または generalization_config.py
                         │
                         v
configs/experiment_profiles.py
                         │
                         v
train.py::_run_training()
                         │
                         ├─ env_factory.py::make_training_env()
                         │       └─ MetaDriveEnv + Monitor
                         │
                         ├─ SubprocVecEnv（既定4 process）
                         │
                         └─ PPO("MlpPolicy", ...)
                                  │
                                  v
                             model.learn()
                                  │
             ┌────────────────────┴────────────────────┐
             v                                         v
  OnPolicyAlgorithm.collect_rollouts()             PPO.train()
             │                                         │
             v                                         v
 obs → Actor/Critic → Action ID              GAE/returnを使ったloss
             │                                         │
             v                                         v
 MetaDrive step → reward/info                 backward → Adam step
             │
             v
 RolloutBuffer
~~~

### 4.2 1 decisionで起きること

1. MetaDriveが現在のObservationを返す。
2. SB3のActorが9 Actionのlogitを計算する。
3. 学習中はCategorical分布からAction IDをsampleする。
4. MetaDriveの `EnvInputPolicy` がAction IDを `[steering, throttle_brake]` に変換する。
5. MetaDriveが物理simulationを5 substep進める。
6. MetaDriveがnext Observation、Reward、terminated、truncated、infoを返す。
7. SB3がObservation、Action、Reward、value、log probabilityなどをRolloutBufferへ保存する。
8. 1 rolloutが貯まるとGAEとreturnを計算し、PPOがActor/Criticを更新する。

MetaDriveの既定値は `physics_world_step_size=0.02` 秒、`decision_repeat=5` である。そのため1 Actionはsimulation上0.1秒、制御周期は10 Hzである。

根拠:

- [train.py](train.py) `:223-374`
- [env_factory.py](env_factory.py) `:18-73`
- [SB3 on_policy_algorithm.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/on_policy_algorithm.py) `:162-268, 300-341`
- [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py) `:435-440, 462-468`
- [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py) `:189-191`

## 5. Actor-Criticとは何か

Actor-Criticは、この文書内だけの便宜的な呼び名ではなく、強化学習で一般に使われる構成である。PPOはActor-Critic型のアルゴリズムであり、SB3でもクラス名が `ActorCriticPolicy` になっている。

### 5.1 Actor

Actorは、状態 `s` におけるAction分布 `πθ(a|s)` を表す。ここで `θ` はActor側の学習parameterである。

この実装ではAction spaceが離散なので、Actorの最終層は9個のlogitを出す。

~~~text
Observation
    ↓
Actor MLP
    ↓
9 logits
    ↓
Categorical distribution
    ↓
Action ID 0～8
~~~

学習時はsampleするため探索が含まれる。評価側の `model.predict(..., deterministic=True)` では、最も確率の高いActionを選ぶ。

### 5.2 Critic

Criticは状態価値を近似する。

$$
V_\phi(s) \approx \mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s\right]
$$

`φ` はCritic側の学習parameterである。Criticはsteeringやthrottleを返さない。実際のreturnと予測値の差からvalue lossを計算し、Actor更新に使うadvantageの基準を提供する。

### 5.3 この実装のネットワーク

`policy_kwargs` を指定していないため、SB3 2.9.0の既定構造が使われる。

| 部分 | 構造 |
| --- | --- |
| feature extractor | `FlattenExtractor`。ベクトルをそのまま渡し、学習parameterはない |
| Actor hidden | `64 → Tanh → 64 → Tanh` |
| Actor output | 9 logits |
| Critic hidden | `64 → Tanh → 64 → Tanh` |
| Critic output | 状態価値1個 |
| optimizer | Adam、`eps=1e-5` |
| initialization | orthogonal initialization |

`official` の構造は次のとおりである。

~~~text
Actor : 259 → 64 → 64 → 9
Critic: 259 → 64 → 64 → 1
~~~

ActorとCriticは同じObservationを受け取るが、既定のMLP hidden layerは別々である。共有されるfeature extractorはparameterを持たないFlatten処理である。

`official` のtrainable parameter数は、ソース上の層構造から次のように導出できる。

| 部分 | parameter数 |
| --- | ---: |
| Actor | `(259×64+64) + (64×64+64) + (64×9+9) = 21,385` |
| Critic | `(259×64+64) + (64×64+64) + (64×1+1) = 20,865` |
| 合計 | **42,250** |

`generalization` は入力が261次元なので合計42,506 parameterになる。

根拠:

- [SB3 policies.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/policies.py) `:416-535, 570-658`
- [SB3 torch_layers.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/torch_layers.py) `:184-261`
- [SB3 distributions.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/distributions.py) `:265-305, 670-696`

### 5.4 学習で更新されるもの・されないもの

更新されるもの:

- Actorの2 hidden layerと9-logit出力層のweight/bias
- Criticの2 hidden layerとvalue出力層のweight/bias
- Adam optimizerの内部状態

更新されないもの:

- Reward関数の係数
- cost関数の係数
- 車両物理parameter
- 道路生成規則
- Action IDからsteering/throttleへの対応
- Observationの計算式
- MetaDriveのtraffic policy

このPPOはworld modelを持たず、将来の画像や車両軌跡を明示的に予測するモデルではない。また、recurrent networkではないため、過去Observationを内部memoryとして保持する構成でもない。

## 6. PPOが収集するデータとGAE

RolloutBufferには次が保存される。

- Observation
- Action
- scalar Reward
- episode start
- Actorが選んだActionのlog probability
- Criticのvalue予測
- 後処理で計算するadvantage
- 後処理で計算するreturn

`cost` フィールドはない。

### 6.1 GAE

SB3のGeneralized Advantage Estimationは、概略として次を計算する。

$$
\delta_t=r_t+\gamma V(s_{t+1})(1-d_t)-V(s_t)
$$

$$
A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}
$$

$$
R_t=A_t+V(s_t)
$$

ここで、

- `d_t`: episode終了を表す値
- `γ=0.99`: discount factor
- `λ=0.95`: GAEのbias/variance調整
- `A_t`: Actor更新に使うadvantage
- `R_t`: Criticの教師信号になるreturn

である。

根拠:

- [SB3 buffers.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/buffers.py) `:366-438`

## 7. PPOのloss

### 7.1 Policy loss

更新前Policyと更新後PolicyのAction確率比を使う。

$$
r_t(\theta)=\exp\left(\log\pi_\theta(a_t\mid s_t)-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)\right)
$$

PPOのclipped surrogate objectiveに対応する実装は次である。

$$
L_{\mathrm{policy}}=-\mathbb{E}\left[\min\left(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right]
$$

この実装では `ε=0.2` である。minibatch内のadvantageは既定で平均0・標準偏差1へ正規化される。

### 7.2 Value loss

$$
L_{\mathrm{value}}=\operatorname{MSE}\left(R_t,V_\phi(s_t)\right)
$$

`clip_range_vf=None` なので、Critic出力の追加clipは行わない。

### 7.3 Entropy lossとtotal loss

SB3はCategorical分布のentropyを計算する。

$$
L_{\mathrm{entropy}}=-\mathbb{E}\left[H(\pi)\right]
$$

total lossは次である。

$$
L=L_{\mathrm{policy}}+\mathrm{ent\_coef}\,L_{\mathrm{entropy}}+\mathrm{vf\_coef}\,L_{\mathrm{value}}
$$

現在の既定値は `ent_coef=0.0`、`vf_coef=0.5` なので、実際にgradientへ寄与する式は次になる。

$$
L=L_{\mathrm{policy}}+0.5\,L_{\mathrm{value}}
$$

entropy自体は計算・記録されるが、係数0なので最適化項としては寄与しない。探索は学習中のCategorical samplingによって存在するが、明示的なentropy bonusはない。

### 7.4 Backpropagation

各minibatchで次を実行する。

1. optimizer gradientを0にする。
2. `loss.backward()` でgradientを計算する。
3. gradient normを0.5でclipする。
4. Adamの `optimizer.step()` でActor/Critic parameterを更新する。

根拠:

- [SB3 ppo.py](.venv/lib/python3.12/site-packages/stable_baselines3/ppo/ppo.py) `:184-299`

## 8. PPO hyperparameterと更新量

### 8.1 profileが明示する値

| parameter | `official` | `generalization` |
| --- | ---: | ---: |
| policy | `MlpPolicy` | `MlpPolicy` |
| RL seed | 0 | 0 |
| environments | 4 | 4 |
| `n_steps` / env | 4,096 | 4,096 |
| requested timesteps | 300,000 | 1,000,000 |
| log interval | 4 | 4 |

根拠:

- [configs/phase0_config.py](configs/phase0_config.py) `:25-33`
- [configs/generalization_config.py](configs/generalization_config.py) `:38-45`

### 8.2 SB3 2.9.0から継承する既定値

| parameter | 値 |
| --- | ---: |
| learning rate | `3e-4` |
| batch size | 64 |
| epochs / rollout | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `clip_range_vf` | `None` |
| normalize advantage | `True` |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| max gradient norm | 0.5 |
| `use_sde` | `False` |
| `target_kl` | `None` |

根拠:

- [SB3 ppo.py](.venv/lib/python3.12/site-packages/stable_baselines3/ppo/ppo.py) `:80-100`

### 8.3 rollout単位への丸め

両profileとも1 rolloutは、

$$
4\ \mathrm{environments}\times 4{,}096\ \mathrm{steps}=16{,}384\ \mathrm{transitions}
$$

である。

| profile | requested | rollout数 | 完走時actual |
| --- | ---: | ---: | ---: |
| `official` | 300,000 | 19 | 311,296 |
| `generalization` | 1,000,000 | 62 | 1,015,808 |

`official` では1 epochあたり `16,384 / 64 = 256` minibatch、1 rolloutあたり10 epochなので2,560 optimizer stepとなる。19 rollout完走時は48,640 optimizer stepである。一方、SB3のログにある `n_updates=190` はoptimizer step数ではなく、19 rollout × 10 epochを数えた値である。

## 9. MetaDriveのReward

`official` と `generalization` はReward設定を上書きしていないため、同じMetaDrive defaultを使う。

| config | 値 |
| --- | ---: |
| `driving_reward` | 1.0 |
| `speed_reward` | 0.1 |
| `success_reward` | 10.0 |
| `out_of_road_penalty` | 5.0 |
| `crash_vehicle_penalty` | 5.0 |
| `crash_object_penalty` | 5.0 |
| `crash_sidewalk_penalty` | 0.0 |
| `use_lateral_reward` | `False` |

根拠:

- [MetaDrive metadrive_env.py](../metadrive/metadrive/envs/metadrive_env.py) `:70-84`

### 9.1 通常stepのdense Reward

$$
r_{\mathrm{dense}}=(long_{\mathrm{now}}-long_{\mathrm{last}})\,lateral_{\mathrm{factor}}\,positive_{\mathrm{road}}+0.1\,\frac{speed_{\mathrm{km/h}}}{max\_speed_{\mathrm{km/h}}}\,positive_{\mathrm{road}}
$$

現在は `use_lateral_reward=False` なので `lateral_factor=1` である。

- 第1項は、現在の参照laneに沿った前進距離をRewardにする。
- 第2項は、最大速度に対する速度比を小さなRewardとして加える。
- 逆向き道路の場合は `positive_road=-1` になり得る。

### 9.2 terminal Rewardは加算ではなく置換

terminal条件では `r_dense` にbonus/penaltyを加えるのではなく、次の値で置き換える。

| 優先順位 | 条件 | 環境が返すReward |
| ---: | --- | ---: |
| 1 | destination到着 | +10 |
| 2 | out of road | -5 |
| 3 | vehicle crash | -5 |
| 4 | object crash | -5 |
| 5 | sidewalk crash | -0 |

`step_info["step_reward"]` は置換前のdense Rewardを保存する。terminal stepで環境が返すRewardと `info["step_reward"]` が異なる可能性がある。

また、既定のout-of-road判定はlane外、連続線、sidewalk crashを含む。そのためsidewalk crashは通常、先にout-of-road分岐へ入り-5になる。building/human crashは終了条件だが、Reward関数にはそれ専用のterminal penalty分岐がない。同じstepで別条件がなければdense Rewardのまま終了し得る。

根拠:

- [MetaDrive metadrive_env.py](../metadrive/metadrive/envs/metadrive_env.py) `:234-290`

## 10. cost関数と「学習されない」理由

MetaDriveのcost係数は次のとおりである。

| config | 値 |
| --- | ---: |
| `out_of_road_cost` | 1.0 |
| `crash_vehicle_cost` | 1.0 |
| `crash_object_cost` | 1.0 |

`cost_function()` は次の優先順位で、加算ではなく1つのcostを返す。

~~~text
out of road      → cost = 1
else vehicle crash → cost = 1
else object crash  → cost = 1
else                cost = 0
~~~

MetaDriveの `BaseEnv._get_step_return()` はcostの数値戻り値を `_` で受けて捨てる一方、cost情報を `info["cost"]` へmergeする。

その後、SB3の `collect_rollouts()` は `infos` を受け取るが、RolloutBufferへ追加するのは次だけである。

~~~text
observation
action
reward
episode_start
value
log_probability
~~~

RolloutBufferにcost fieldはなく、PPO lossにもcost項はない。したがって結論は次である。

> MetaDriveはcostを診断情報として計算しているが、現在のモデルはcostを最適化していない。RewardのみでActor/Criticを学習している。

さらに、学習環境の `Monitor` は `info_keywords` を指定しておらず、現行評価telemetryもcostをstep集計していない。そのため現在のMonitor CSVやevaluation JSONから平均cost曲線を復元することもできない。

根拠:

- [MetaDrive metadrive_env.py](../metadrive/metadrive/envs/metadrive_env.py) `:206-216`
- [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py) `:606-641`
- [SB3 on_policy_algorithm.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/on_policy_algorithm.py) `:218-254`
- [SB3 buffers.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/buffers.py) `:366-477`
- [env_factory.py](env_factory.py) `:34-73`

## 11. terminatedとtruncated

MetaDrive defaultで有効な主なtask terminationは次である。

- destination到着
- out of road
- continuous lane lineへの侵入
- vehicle crash
- object crash
- human crash
- building crash
- sidewalk crashはout-of-road判定にも含まれる

既定では、

~~~text
out_of_route_done      = False
out_of_road_done       = True
on_continuous_line_done = True
on_broken_line_done    = False
crash_vehicle_done     = True
crash_object_done      = True
crash_human_done       = True
truncate_as_terminate  = False
~~~

`horizon` 到達時は `max_step=True` になり、通常は `truncated=True`、`terminated=False` になる。同じstepで衝突や到着も起きた場合は両方がtrueになり得る。

SB3のVecEnvは `terminated or truncated` をepisode終了として扱う。ただし純粋なtime limit truncationではterminal Observationのvalueを使ってRewardへ `γV(s_T)` を加え、bootstrapする。

| profile | horizon | 最大simulation時間の目安 |
| --- | ---: | ---: |
| `official` | 500 decision | 約50秒 |
| `generalization` | 1,000 decision | 約100秒 |

根拠:

- [MetaDrive metadrive_env.py](../metadrive/metadrive/envs/metadrive_env.py) `:86-94, 133-204`
- [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py) `:82-86, 623-632`
- [SB3 on_policy_algorithm.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/on_policy_algorithm.py) `:234-245`

## 12. `official` と `generalization`

### 12.1 設定比較

| 項目 | `official` | `generalization` |
| --- | --- | --- |
| map | `"C"` | `3` |
| train scenario seed | `[5, 6)` | `[1000, 2000)` |
| evaluation seed | `[5, 6)` | `[0, 200)` |
| horizon | 500 | 1,000 |
| spawn lane | 固定 | random |
| lane width | 3.5 m | `[3.0, 4.5)` からrandom |
| lane count | 3 | 2または3 |
| ego model | default固定 | S/M/L/XL/defaultの5種から等確率 |
| traffic density | 0 | 0.1 |
| traffic randomization | trafficなし | `random_traffic=False`。scenario seedに従う |
| accident probability | 0 | 0 |
| map cache | default `True` | `False` |
| Observation | 259 | 261 |
| Action | `Discrete(9)` | `Discrete(9)` |
| Reward/cost | MetaDrive default | MetaDrive default |

根拠:

- [configs/phase0_config.py](configs/phase0_config.py) `:11-33`
- [configs/generalization_config.py](configs/generalization_config.py) `:8-47`
- [MetaDrive base_map.py](../metadrive/metadrive/component/map/base_map.py) `:37-41`
- [MetaDrive pg_map_manager.py](../metadrive/metadrive/manager/pg_map_manager.py) `:68-74`
- [MetaDrive vehicle_type.py](../metadrive/metadrive/component/vehicle/vehicle_type.py) `:421-433`

### 12.2 道路生成

`official` の `map="C"` は、FirstPGBlockにCurve blockを1つ接続するblock sequenceである。Curveのparameter sampling範囲は次である。

| parameter | 範囲 |
| --- | --- |
| length | 40～80 m |
| radius | 25～60 m |
| angle | 45～135° |
| direction | 2方向 |

scenario seed 5で具体的なgeometryは決まるが、本調査では実現値を個別計算していない。

`generalization` の `map=3` は、FirstPGBlockに3つのblockを確率的に接続する。標準v2分布は次である。

| block | 確率 |
| --- | ---: |
| Curve | 0.30 |
| Straight | 0.10 |
| InRampOnStraight | 0.10 |
| OutRampOnStraight | 0.10 |
| StdInterSection | 0.15 |
| StdTInterSection | 0.15 |
| Roundabout | 0.10 |

`traffic_density=0.1` は「trafficなし」ではない。Trigger modeのIDM trafficが生成される。`random_traffic=False` は、traffic配置をscenario seedに対応させる設定である。

### 12.3 seedの注意

`train.py` は `set_random_seed(args.seed)` を呼ぶが、PPO constructorの `seed` 引数は意図的に省略している。PPOへ `seed` を渡すと、SB3はVecEnvへseedを設定し、次回reset時にworkerごとのseedをMetaDriveの `reset(seed=...)` へ伝播する。MetaDrive 0.4.3はその値を単なるRL乱数seedではなくscenario indexとして解釈するためである。

`official` はscenarioが1個なので問題が表面化しにくい。一方、`generalization` のreset時scenario順序は、RL seed 0だけから完全再現できるとは限らない。training metadataにもこの区別を記録する設計になっている。

### 12.4 model互換性

`generalization` は車長・車幅の2特徴を追加するため、`official` の259入力modelとshapeが合わない。同じAction/Rewardでもmodelを共有できない。

## 13. Observationの生成経路

`official` はObservation設定を上書きしない。

~~~text
BaseEnv.get_single_observation()
  image_observation = False
  agent_observation = None
          │
          v
LidarStateObservation
  ├─ StateObservation
  │    ├─ Ego/road state
  │    └─ Navigation
  ├─ optional surrounding vehicle features
  └─ LiDAR cloud points
~~~

既定sensor設定は次である。

~~~python
lidar = {
    "num_lasers": 240,
    "distance": 50,
    "num_others": 0,
    "gaussian_noise": 0.0,
    "dropout_prob": 0.0,
    "add_others_navi": False,
}
side_detector = {"num_lasers": 0, ...}
lane_line_detector = {"num_lasers": 0, ...}
~~~

根拠:

- [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py) `:71-80, 168-176, 667-679`
- [MetaDrive state_obs.py](../metadrive/metadrive/obs/state_obs.py)

## 14. `official` の259次元

### 14.1 shapeの導出

~~~text
StateObservation:
  base ego state                    6
  Navigation                       10
  side detector無効時の代替値       2
  lane-line detector無効時の代替値  1
                                   --
                                   19

LidarStateObservation:
  LiDAR rays                      240
  num_others × 4                    0
  other navigation                  0
                                  ---
                                  240

合計: 19 + 240 = 259
~~~

全体indexは次のとおりである。

| 0-based index | 次元 | 内容 |
| ---: | ---: | --- |
| 0～8 | 9 | Ego・道路状態 |
| 9～13 | 5 | checkpoint 1 |
| 14～18 | 5 | checkpoint 2 |
| 19～258 | 240 | LiDAR |
| — | 0 | 周辺車両の構造化追加特徴 |

### 14.2 Ego・道路状態9次元

`clip` は0～1へのclipを表す。

| index | 正確な内容 | policyへ渡る式・値 |
| ---: | --- | --- |
| 0 | route左端までの横方向距離 | `clip(d_left / 18)` |
| 1 | route右端までの横方向距離 | `clip(d_right / 18)` |
| 2 | laneに対するheading encoding | lane横方向ベクトルとvehicle headingの正規化内積を `[-1,1]→[0,1]` |
| 3 | 速度encoding | `clip((speed_km_h + 1) / (max_speed_km_h + 1))` |
| 4 | 現在steering encoding | `clip((vehicle.steering / 60 + 1) / 2)` |
| 5 | 最新Actionのsteering | `clip((action[0] + 1) / 2)` |
| 6 | 最新Actionのthrottle/brake | `clip((action[1] + 1) / 2)` |
| 7 | 符号なしyaw-rate encoding | `clip(arccos(clip(cos_beta, 0, 1)) / 0.1)` |
| 8 | lane-local横位置 | `clip(0.5 + lateral / 4.5)` |

#### index 0、1

これは現在lane一本の左右端ではなく、route全体の左右端までの横方向距離である。分母18 mは、

$$
\left(\mathrm{MAX\_LANE\_NUM}+1\right)\times\mathrm{MAX\_LANE\_WIDTH}=(3+1)\times4.5=18
$$

から来る。`official` の通常道路幅は3 lane × 3.5 m = 10.5 mだが、正規化分母は固定上限18 mである。

#### index 2

`heading_diff` という名前だが、角度差そのものではない。lane横方向ベクトルとvehicle headingの内積を0～1へ移した値で、lane方向と一致するとおおむね0.5になる。

#### index 3

既定最大速度は80 km/hなので、停止時でも0ではなく `1/81 ≈ 0.01235`、80 km/hで1になる。

#### index 4

`vehicle.steering` はAction由来の `[-1,1]` だが、観測側は定数 `MAX_STEERING=60` で割る。そのため `official` の離散steeringでは値が約0.49167、0.5、0.50833という狭い範囲になる。これは現在のMetaDrive source上の実装上の特徴である。

#### index 5、6

実装順はlatest steering、latest throttle/brakeである。`state_obs.py` のdocstringは逆順に読めるため、docstringよりリストappend順とAction実装を優先する。

#### index 7

左右の符号は消える。heading変化が0.1 rad/decision以上になると1へ飽和する。ソースコメントの「angular acceleration」より、符号なし・上限制限付きyaw-rate encodingと解釈する方が実装に近い。

#### index 8

lane中心のraw lateralは0だが、policy入力は0.5である。

根拠:

- [MetaDrive state_obs.py](../metadrive/metadrive/obs/state_obs.py) `:67-162`
- [MetaDrive base_vehicle.py](../metadrive/metadrive/component/vehicle/base_vehicle.py) `:77-79, 210-230, 472-480, 524-536, 565-589`
- [MetaDrive base_map.py](../metadrive/metadrive/component/map/base_map.py) `:37-41`

## 15. Navigation 10次元

Navigationは2 checkpoint × 5要素である。

| 全体index | 対象 | 内容 |
| ---: | --- | --- |
| 9 | checkpoint 1 | ego heading方向の相対位置 |
| 10 | checkpoint 1 | ego right-hand-side方向の相対位置 |
| 11 | checkpoint 1 | laneの曲率半径encoding |
| 12 | checkpoint 1 | 曲がる方向encoding |
| 13 | checkpoint 1 | laneの総曲がり角encoding |
| 14 | checkpoint 2 | ego heading方向の相対位置 |
| 15 | checkpoint 2 | ego right-hand-side方向の相対位置 |
| 16 | checkpoint 2 | laneの曲率半径encoding |
| 17 | checkpoint 2 | 曲がる方向encoding |
| 18 | checkpoint 2 | laneの総曲がり角encoding |

checkpoint 1はcurrent roadのlane終端中央、checkpoint 2はnext roadのlane終端中央を表す。最終roadでnext roadがない場合、checkpoint 2はcurrent laneを再利用する。

### 15.1 相対位置

vehicleからcheckpointへのベクトルが50 mを超える場合、方向を保って長さ50 mへcropする。その後、車両前後軸と右軸へ射影する。

$$
forward^{\prime}=\operatorname{clip}\left(\frac{forward/50+1}{2},0,1\right)
$$

$$
right^{\prime}=\operatorname{clip}\left(\frac{right/50+1}{2},0,1\right)
$$

各軸で-50 mは0、0 mは0.5、+50 mは1になる。

### 15.2 曲率半径

CircularLaneでは、

$$
radius^{\prime}=\operatorname{clip}\left(\frac{radius}{60+lane_{num}\times lane_{width}},0,1\right)
$$

であり、直線では0になる。`official` の3 lane、3.5 mでは分母70.5である。

### 15.3 曲がる方向

| lane | 観測値 |
| --- | ---: |
| clockwise | 1.0 |
| anticlockwise | 0.0 |
| straight | 0.5 |

### 15.4 総曲がり角

$$
angle^{\prime}=\operatorname{clip}\left(\frac{angle_{degree}/135+1}{2},0,1\right)
$$

| lane | 観測値 |
| --- | ---: |
| straight | 0.5 |
| 45° | 約0.667 |
| 90° | 約0.833 |
| 135° | 1.0 |

docstringの「straightは方向・角度0」は正規化前のraw値を指す。最終的にPolicyへ渡る値はどちらも0.5である。

根拠:

- [MetaDrive base_navigation.py](../metadrive/metadrive/component/navigation_module/base_navigation.py) `:18-20, 280-282`
- [MetaDrive node_network_navigation.py](../metadrive/metadrive/component/navigation_module/node_network_navigation.py) `:189-201, 288-347`

## 16. LiDAR 240次元

LiDAR indexを `i=0,...,239` とすると、全体Observationのindexは `19+i` である。

$$
\theta_i=\mathrm{vehicle.heading\_theta}+i\frac{2\pi}{240}
$$

240本で360°を一周するため、角度間隔は1.5°である。`i=0` はvehicle正面、`i=120` は後方である。docstringにはclockwiseとあるが、厳密な定義は上の加算式を優先する。

### 16.1 値の意味

LiDARは高さ1.2 mから最大50 mのrayを飛ばし、最初に衝突した表面までのBullet hit fractionを返す。

$$
o_{19+i}=\frac{d_i}{50}
$$

| ray上の状態 | LiDAR値 |
| --- | ---: |
| 5 mでhit | 約0.1 |
| 10 mでhit | 約0.2 |
| 25 mでhit | 約0.5 |
| 50 mでhit | 約1.0 |
| hitなし | 1.0 |
| broad-phase maskでrayを省略 | 1.0 |

したがって、LiDAR値は「近さ」ではない。**小さいほど近く、1.0は最大距離または未検出**である。物体中心までではなく、最初に当たったcollision surfaceまでの距離である。

### 16.2 検出対象

LiDAR maskは次を対象にする。

- Vehicle
- InvisibleWall
- TrafficObject
- TrafficParticipants

通常のlane surface、lane line、sidewalkはこのmaskに含まれない。道路境界情報はObservation index 0、1から別途与えられる。

`num_others=0` は、検出車両を別枠の相対位置・相対速度4特徴として追加しないという意味であり、LiDAR rayがvehicleにhitしないという意味ではない。

`official` は `gaussian_noise=0`、`dropout_prob=0` なので、ray値はnoise/dropoutで変更されない。

根拠:

- [MetaDrive state_obs.py](../metadrive/metadrive/obs/state_obs.py) `:165-247`
- [MetaDrive distance_detector.py](../metadrive/metadrive/component/sensors/distance_detector.py) `:27-85, 117-159, 176-179`
- [MetaDrive lidar.py](../metadrive/metadrive/component/sensors/lidar.py) `:16-31`
- [MetaDrive constants.py](../metadrive/metadrive/constants.py) `:244-246`
- [MetaDrive math.py](../metadrive/metadrive/utils/math.py) `:75-80`

## 17. LiDAR最大距離を変更する

最大50 mは固定仕様ではなく、MetaDrive default configの値である。nested configで100 mなどへ変更できる。

~~~python
CUSTOM_ENV_CONFIG = {
    **OFFICIAL_ENV_CONFIG,
    "vehicle_config": {
        "lidar": {
            "distance": 100,
        },
    },
}
~~~

MetaDriveの `Config.update()` はnested dictを再帰mergeするため、`num_lasers` など未指定のLiDAR設定はdefaultのまま残る。

100 m設定では、

$$
o_i=\frac{d_i}{100}
$$

となる。

| 距離 | 50 m設定 | 100 m設定 |
| ---: | ---: | ---: |
| 10 m | 0.2 | 0.1 |
| 25 m | 0.5 | 0.25 |
| 50 m | 1.0 | 0.5 |
| 100 m | 範囲外 | 1.0 |
| hitなし | 1.0 | 1.0 |

`distance` だけを変えてもray本数は240のままなので、Observation shapeは259のままである。`num_lasers` を変えるとshapeも変わる。

shapeが同じでも入力の意味は変わるため、50 mで学習したmodelを100 mで評価するのは通常の同一条件評価ではなく、sensor range変更に対する汎化実験である。基本的には学習・評価で距離を揃えるか、複数距離を含めて再学習する。

`OFFICIAL_ENV_CONFIG` は公式例再現用にtestでhash固定されている。直接変更すると `official` contractを壊すため、別config/profileとして追加する方がよい。

Navigationにも `NAVI_POINT_DIST=50` があるが、これはcheckpointベクトルのcrop距離で、LiDARの `distance` とは独立である。

根拠:

- [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py) `:168-173, 288-297`
- [MetaDrive config.py](../metadrive/metadrive/utils/config.py) `:126-180`
- [MetaDrive base_navigation.py](../metadrive/metadrive/component/navigation_module/base_navigation.py) `:18-20`
- [tests/test_phase0_contract.py](tests/test_phase0_contract.py) `:15-30`

## 18. `generalization` の261次元

`random_agent_model=True` により、Observationの先頭へ次の2要素を追加する。

| index | 内容 |
| ---: | --- |
| 0 | `vehicle.LENGTH / 10` |
| 1 | `vehicle.WIDTH / 2.5` |

そのため全体indexは次になる。

| 0-based index | 内容 |
| ---: | --- |
| 0～1 | ego vehicle length/width |
| 2～10 | `official` のEgo・道路状態9要素 |
| 11～20 | Navigation 10要素 |
| 21～260 | LiDAR 240要素 |

MetaDriveのコメントには「4 types」とあるが、実装上の候補はS/M/L/XL/defaultの5種であり、指定確率がなければ等確率である。

根拠:

- [MetaDrive state_obs.py](../metadrive/metadrive/obs/state_obs.py) `:22-28, 71-78`
- [MetaDrive vehicle_type.py](../metadrive/metadrive/component/vehicle/vehicle_type.py) `:421-433`

## 19. 「正規化」の正確な意味

Observation spaceは `Box(0, 1, shape=(259,), dtype=float32)` であり、多くの値は個別の式で0～1へ変換・clipされる。ただし、機械学習でいう平均0・分散1の標準化ではない。

特に次へ注意する。

- 停止速度は0ではなく約0.01235。
- lane中心の横位置は0.5。
- laneとheadingが一致したときのheading encodingもおおむね0.5。
- straight laneのNavigation direction/angleは0.5。
- LiDARの1.0は最大距離hitと未検出の両方。
- current steering encodingは0.5付近の狭い範囲。
- `VecNormalize` は使っていない。

したがって、「すべて正規化された」とは、次のように読むのが正確である。

> 259個の各特徴が、それぞれ固有の式で主に0～1へ符号化・clipされたベクトルであり、物理量の生値やz-scoreではない。

## 20. Action space `Discrete(9)`

`discrete_steering_dim=3`、`discrete_throttle_dim=3` の直積を、単一のAction IDにしている。

~~~python
steering = (action_id % 3) - 1
throttle_brake = (action_id // 3) - 1
~~~

完全な対応は次である。

| Action ID | steering | 操舵 | throttle/brake | 前後操作 |
| ---: | ---: | --- | ---: | --- |
| 0 | -1 | 右 | -1 | brake |
| 1 | 0 | 直進 | -1 | brake |
| 2 | +1 | 左 | -1 | brake |
| 3 | -1 | 右 | 0 | neutral |
| 4 | 0 | 直進 | 0 | neutral |
| 5 | +1 | 左 | 0 | neutral |
| 6 | -1 | 右 | +1 | throttle |
| 7 | 0 | 直進 | +1 | throttle |
| 8 | +1 | 左 | +1 | throttle |

steeringの符号は、vehicle進行方向を向いた運転者の視点で次になる。

~~~text
+1 = 左
 0 = 直進
-1 = 右
~~~

根拠は3経路で一致する。

1. Manual controllerはA/leftでsteeringを加算し、D/rightで減算する。
2. MetaDrive testは `env.step([+0.8,+0.8])` をleft、`[-0.8,+0.8]` をrightとして、headingとpositionの変化までassertする。
3. RL projectのevaluation decoderも正をLEFT、負をRIGHTとする。

`official` は `enable_reverse=False` なので、`throttle_brake=-1` はreverseではなくbrakeである。停止中やbrake中にsteeringを指定しても、車両がただちに横へ移動するという意味ではない。

根拠:

- [MetaDrive env_input_policy.py](../metadrive/metadrive/policy/env_input_policy.py) `:40-68`
- [MetaDrive base_vehicle.py](../metadrive/metadrive/component/vehicle/base_vehicle.py) `:472-520`
- [MetaDrive manual_controller.py](../metadrive/metadrive/engine/core/manual_controller.py) `:45-102`
- [MetaDrive test_set_get_vehicle_attribute.py](../metadrive/metadrive/tests/test_component/test_set_get_vehicle_attribute.py) `:120-142`

## 21. 学習と評価の違い

| 項目 | 学習 | 評価 |
| --- | --- | --- |
| model | `PPO(...)` で生成 | `PPO.load(...)` |
| Action選択 | Categoricalからsample | `deterministic=True` |
| Reward | GAE、return、lossへ使う | episode指標へ集計するだけ |
| Critic | valueを学習 | 重みを更新しない |
| rollout buffer | 使う | 使わない |
| backward/optimizer | あり | なし |
| model重み | 更新される | 変更されない |
| environments | 既定4 process | 1 environment |

評価で高いRewardを得ても、その結果が自動的にmodelへ戻って追加学習されることはない。

## 22. 現在の成果物と確認できる実績

### 22.1 snapshotが変化したこと

このチャット前半の最初のartifact確認時点では、`models/`、`logs/`、`outputs/` は実質 `.gitkeep` のみで、学習済みmodelや評価JSONを確認できなかった。

本書作成時点で再確認すると、workspaceには `official` のmodel、学習log、Monitor、TensorBoard、評価JSON/JSONL/GIF/MP4/PNGが追加されている。そのため、以下は**本書作成時点の現存artifact**を根拠に更新した内容である。

### 22.2 `official` model

| 項目 | 現在の確認値 |
| --- | --- |
| model | `models/phase0_official.zip` |
| size | 552,645 bytes |
| SHA-256 | `a9b5b1f92711e312af51c5a8ed6f827992fa702a643ca1471359e9209dc3d6de` |
| SB3 version | 2.9.0 |
| `num_timesteps` | 311,296 |
| requested total | 300,000 |
| `_n_updates` | 190 |
| Observation space | `Box(0,1,(259,),float32)` |
| Action space | `Discrete(9)` |
| environments | 4 |
| `n_steps` | 4,096 |

`num_timesteps`、`_n_updates`、spacesはzip内のSB3 `data` から直接確認した。zipには `policy.pth` だけでなく `policy.optimizer.pth` も含まれる。

[logs/full_train.log](logs/full_train.log) は `model.learn()` 後の保存、model SHA、reload成功を記録している。ログ表示は `log_interval=4` のため262,144 timestepsが最後の表だが、model archive内の `num_timesteps=311296` が完走後の直接証拠である。

一方、ログが参照する旧path `outputs/phase0_official_training_metadata.json` は現在存在しない。現行 `train.py` は `outputs/official/training/<model-name>/training_metadata.json` を出す設計だが、今回存在するmodelのrunはその構成より前のものと見られる。したがって、model本体とlogは確認できるが、現行schemaのtraining metadata一式は欠けている。

### 22.3 `official` evaluation

[outputs/official/evaluation/phase0_official/evaluation.json](outputs/official/evaluation/phase0_official/evaluation.json) から確認できる1 episodeの結果は次である。

| 項目 | 値 |
| --- | ---: |
| scenario seed | 5 |
| episode count | 1 |
| success | 1 |
| total Reward | 169.50275029569968 |
| episode length | 127 decision |
| simulation時間 | 12.7秒 |
| route completion | 0.976623203647132 |
| out of road | false |
| crash | false |

これは学習と同じ単一scenario seed 5でのdeterministic評価である。「このepisodeを成功した」ことは確認できるが、未知mapやtrafficへの汎化性能を示す結果ではない。1 episodeのsuccess rate 1.0を一般性能保証として扱ってはいけない。

### 22.4 `generalization`

本書作成時点で、`generalization` のfull model、training metadata、Monitor/TensorBoard、evaluation JSONは存在しない。

[RUN_REPORT.md](RUN_REPORT.md) は「4 workerで256 timestep」のgeneralization smoke、保存・再読込、未見seed 0～4評価、JSON生成を完了したと報告している。ただし、記載上の256が `n_steps`、requested total、actual totalのどれを意味するかは特定できない。そのsmoke artifactも現在のworkspaceに残っていないため、詳細はreported情報であり、full 1,000,000 timestep学習の証拠ではない。

また、同レポートにあるPPO 64-stepやpytest結果も、今回再実行したのではなくreportedとして扱う。

## 23. ソースを追う推奨順序

次の順に読むと、「設定した値がどこで生成され、変換され、消費されるか」を追いやすい。

### Step 1: profileを確定する

1. [configs/phase0_config.py](configs/phase0_config.py)
2. [configs/generalization_config.py](configs/generalization_config.py)
3. [configs/experiment_profiles.py](configs/experiment_profiles.py)

ここで次を記録する。

~~~text
環境config
PPOに明示するconfig
train/evaluation scenario範囲
Observation shapeを変える設定
~~~

### Step 2: 自作entry pointを追う

4. [train.py](train.py) `main() → _run_training()`
5. [env_factory.py](env_factory.py) `make_training_env()`
6. [evaluate.py](evaluate.py) `main() → evaluation loop`

ここで、自作コードはPPO内部lossやMetaDrive taskを再実装していないことを確認する。

### Step 3: `model.learn()` の中へ入る

7. [SB3 on_policy_algorithm.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/on_policy_algorithm.py)
   - `learn()`
   - `collect_rollouts()`
8. [SB3 buffers.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/buffers.py)
   - `RolloutBuffer.add()`
   - `compute_returns_and_advantage()`
9. [SB3 ppo.py](.venv/lib/python3.12/site-packages/stable_baselines3/ppo/ppo.py)
   - `PPO.__init__()`
   - `PPO.train()`
10. [SB3 policies.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/policies.py)
    - `ActorCriticPolicy`
11. [SB3 distributions.py](.venv/lib/python3.12/site-packages/stable_baselines3/common/distributions.py)
    - `CategoricalDistribution`

この順で、次のdata boundaryを見る。

~~~text
生成: policy(obs) → action/value/log_prob
変換: env.step(action) → reward/next_obs/done/info
保存: RolloutBuffer
後処理: GAE/return
消費: PPO loss
更新: backward/optimizer.step
~~~

### Step 4: MetaDriveの1 stepを追う

12. [MetaDrive base_env.py](../metadrive/metadrive/envs/base_env.py)
    - `step()`
    - `_step_simulator()`
    - `_get_step_return()`
13. [MetaDrive metadrive_env.py](../metadrive/metadrive/envs/metadrive_env.py)
    - `reward_function()`
    - `cost_function()`
    - `done_function()`

`reward` と `info["cost"]` が別経路で作られ、PPO側にはRewardだけが入ることを確認する。

### Step 5: Observationを追う

14. [MetaDrive state_obs.py](../metadrive/metadrive/obs/state_obs.py)
15. [MetaDrive node_network_navigation.py](../metadrive/metadrive/component/navigation_module/node_network_navigation.py)
16. [MetaDrive distance_detector.py](../metadrive/metadrive/component/sensors/distance_detector.py)
17. [MetaDrive lidar.py](../metadrive/metadrive/component/sensors/lidar.py)
18. [MetaDrive base_vehicle.py](../metadrive/metadrive/component/vehicle/base_vehicle.py)

shapeの算術だけで終わらず、`observe()` のappend順を追ってindex順を確定する。

### Step 6: Actionを追う

19. [MetaDrive env_input_policy.py](../metadrive/metadrive/policy/env_input_policy.py)
20. [MetaDrive base_vehicle.py](../metadrive/metadrive/component/vehicle/base_vehicle.py)
21. [MetaDrive manual_controller.py](../metadrive/metadrive/engine/core/manual_controller.py)
22. [MetaDrive steering test](../metadrive/metadrive/tests/test_component/test_set_get_vehicle_attribute.py)

`Action ID → steering/throttle → BulletVehicle` を追い、左右はmanual inputと物理testで照合する。

### Step 7: artifactとcontractを確認する

23. [tests](tests)
24. [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md)
25. [RUN_REPORT.md](RUN_REPORT.md)
26. `models/*.zip` のSB3 `data`
27. `outputs/**/training_metadata.json`
28. `outputs/**/evaluation.json`

設定値、test、reported run、現物artifactを混同しない。

## 24. 読むときのチェックリスト

各値について、次の5点を記録すると迷いにくい。

1. **どこでdefaultが定義されるか**
2. **profileが上書きするか**
3. **runtime configへどうmergeされるか**
4. **どの関数がその値を消費するか**
5. **最終的にrollout/loss/artifactへ残るか**

例えばcostなら次になる。

~~~text
default定義:
  metadrive_env.py の cost係数

生成:
  MetaDriveEnv.cost_function()

変換:
  BaseEnv._get_step_return() が infoへmerge

消費:
  評価・診断から参照可能

学習:
  RolloutBufferにfieldがないためPPO lossでは未使用
~~~

## 25. ソース上の注意点

### 25.1 docstringと実装が一致しない箇所

- `state_obs.py` のAction説明順は実装と逆に読める。実装はsteering、throttle/brake。
- `state_obs.py` の `9 + 10 + 16 + 240` は一般例であり、`official` は `num_others=0` なので16次元は存在しない。
- yaw featureのコメントはangular accelerationだが、実装は符号なしheading変化量/0.1。
- Navigationのstraight direction/angleはraw 0だが、最終encodingは0.5。
- LiDARの「clockwise」はコメントよりangle生成式を優先する。
- `random_agent_model` のコメントは4車種だが、実装候補は5種。

### 25.2 誤解しやすい点

- `num_others=0` は周辺車両が存在しないという意味ではない。
- `traffic_density=0.1` では実際にtrafficが生成される。
- `random_traffic=False` はtrafficなしという意味ではない。
- Observationの0～1化はz-score標準化ではない。
- LiDARの1.0は「遠い」と「未検出」を区別しない。
- Rewardのterminal値はdense Rewardへの加算ではなく置換。
- `info["cost"]` が存在してもPPOがcostを学習しているとは限らない。
- `total_timesteps` はrollout境界で上回る。
- evaluation成功1件は未知scenarioへの汎化保証ではない。

## 26. 最小の再確認コマンド

重い学習を行わずに、sourceとartifactを再確認する例を示す。

~~~bash
# profile設定
nl -ba configs/phase0_config.py
nl -ba configs/generalization_config.py

# model.learn()の呼び出し
rg -n "PPO\\(|model\\.learn" train.py

# Reward / cost / termination
rg -n "def reward_function|def cost_function|def done_function" \
  ../metadrive/metadrive/envs/metadrive_env.py

# Observation
rg -n "class StateObservation|class LidarStateObservation" \
  ../metadrive/metadrive/obs/state_obs.py

# Action変換
rg -n "convert_to_continuous_action" \
  ../metadrive/metadrive/policy/env_input_policy.py

# PPO loss
rg -n "policy_loss|value_loss|entropy_loss|loss.backward" \
  .venv/lib/python3.12/site-packages/stable_baselines3/ppo/ppo.py

# model archive内のmetadata
unzip -p models/phase0_official.zip data |
  rg '"num_timesteps"|"n_steps"|"n_envs"|"batch_size"|"n_epochs"'

# 現在の主要artifact
find models logs outputs -type f \
  ! -path '*/frames/*' -print
~~~

## 27. 調査範囲と未検証事項

今回の文書作成では、source、設定、既存model archive、log、evaluation artifactを読み取り確認した。重い学習や長時間simulationは新たに実行していない。

未検証または限定的な事項:

- `generalization` のfull 1,000,000 timestep学習と性能
- `generalization` のscenario出現順序と全1,000 scenarioの実coverage
- `official` Curve seed 5の具体的なsample済みgeometry parameter
- LiDAR距離100 mへ変更したruntime性能と計算負荷
- cost制約を実際にlossへ導入した場合の性能
- `official` modelの複数episode・未知scenario・traffic条件での性能

既存modelの単一scenario成功はconfirmedだが、それ以上の一般化主張は行わない。

## 28. 関連資料

- [CODE_WALKTHROUGH.md](CODE_WALKTHROUGH.md): 各ファイル・関数の責任分担を広く追う索引
- [RUN_REPORT.md](RUN_REPORT.md): 既存環境と過去実行結果のreported記録
- [README.md](README.md): setup、標準コマンド、profile利用方法
- [評価結果](outputs/official/evaluation/phase0_official/evaluation.json): 現存する`official` 1 episode評価

本書は、上記資料のうち特に「何を学習するか」「どのlossを使うか」「Rewardとcostのどちらが学習へ入るか」「Observation/Actionの全意味」を、実ソースに沿って一か所へ統合したものである。
