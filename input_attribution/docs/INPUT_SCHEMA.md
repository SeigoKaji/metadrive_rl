# 入力schema（MetaDrive標準259）

この文書と `input_attribution.schema` が入力対応表の正本です。標準schemaは、MetaDrive 0.4.3 の `StateObservation.vehicle_state()` と `NodeNetworkNavigation._get_info_for_checkpoint()` が作る19要素に、`LidarStateObservation` が追加する240本を順番どおり連結した `float32[259]` です。実行時にはmodel、実観測のshape、保存したschemaの3つを照合し、Dだけを理由にpadding、切り捨て、順序推定をしません。

## 標準259の先頭19要素

各行には `index`、正確な `name`、`normalization`、`normal_range`、`zero_meaning`、`one_meaning`、`source`、`source_version`、`source_commit`、`group` が入ります。次の式は実装の式を省略せず要約したものです。`clip` は両端を `[0, 1]` に収めます。

| index | name | 実装式の要点 | 0の意味 | 1の意味 |
|---:|---|---|---|---|
| 0 | `road_edge_left_distance_norm` | `clip(dist_to_left_side / ((MAX_LANE_NUM+1)*MAX_LANE_WIDTH), 0, 1)` | 左端距離が0以下 | map幅以上（既定値なら18m以上） |
| 1 | `road_edge_right_distance_norm` | 右端距離を同じmap幅で割る | 右端距離が0以下 | map幅以上（既定値なら18m以上） |
| 2 | `lane_heading_side_component` | `clip(dot(vehicle.heading, right_side_normal), -1, 1)/2+0.5` | dot = -1 | dot = +1 |
| 3 | `speed_km_h_norm` | `clip((speed_km_h+1)/(max_speed_km_h+1), 0, 1)` | raw speed が -1km/h 以下（物理的な停止は厳密な0にならない） | max speed 以上 |
| 4 | `steering_state_norm` | `clip((steering / BaseVehicle.MAX_STEERING + 1)/2, 0, 1)` | -`MAX_STEERING` 以下 | +`MAX_STEERING` 以上 |
| 5 | `last_action_steering_norm` | `clip((last_current_action[1][0]+1)/2, 0, 1)` | 最新保存steeringが -1 以下 | 最新保存steeringが +1 以上 |
| 6 | `last_action_throttle_brake_norm` | `clip((last_current_action[1][1]+1)/2, 0, 1)` | 最新保存throttle/brakeが -1 以下 | 最新保存throttle/brakeが +1 以上 |
| 7 | `yaw_rate_norm` | `clip(arccos(clip(dot(now,last)/(norm(now)*norm(last)),0,1))/0.1, 0, 1)` | heading変化なし | 0.1rad/0.1s以上（符号なし） |
| 8 | `lane_lateral_position_norm` | `clip((local_lateral*2/MAX_LANE_WIDTH+1)/2, 0, 1)` | -`MAX_LANE_WIDTH/2` 以下 | +`MAX_LANE_WIDTH/2` 以上 |
| 9--13 | `navigation_next_checkpoint_*` | heading/right projection は `clip((projection/50+1)/2,0,1)`、radius・direction・angleを後述の設定値でclip | 各列の原値が下限側 | 各列の原値が上限側 |
| 14--18 | `navigation_next_next_checkpoint_*` | 9--13と同じ5列を2番目のcheckpointへ適用 | 各列の原値が下限側 | 各列の原値が上限側 |

navigationの5列は順に `forward_norm`、`right_norm`、`curve_radius_norm`、`curve_direction_norm`、`curve_angle_norm` です。radiusはCircularLaneだけを `ref_lane.radius / (curve_radius_max + lane_count*lane_width)` で正規化し、直線では0です。directionは `dir=-1`（anticlockwise）、0（straight）、+1（clockwise）を `(dir+1)/2` に通します。したがって直線の実値は0.5です。angleは度へ変換して135で割り、直線の実値は同じく0.5です。コメントと実行式が異なる箇所は実行式を採用しています。

## LiDAR 240本

LiDARは観測index 19 から始まり、`beam_index = 0..239` の各行を生成します。共通式は次のとおりです。

```text
radian_unit = 2*pi/240
ray_angle = vehicle.heading_theta + beam_index*radian_unit + start_phase_offset
endpoint = (distance*cos(ray_angle)+x, distance*sin(ray_angle)+y)
cloud_points[beam_index] = hit_fraction
```

標準設定の `start_phase_offset=0`、`distance=50m` では角度刻みは1.5度です。index 0は車両heading方向、beam 60はheading +90度、beam 120は反対、beam 180はheading -90度です。実装コメントには「clockwise」と書かれた版がありますが、ここでは実行式の角度増分と座標変換を検証し、+90度側をMetaDriveのlocal right-side成分として扱います。方向を別のhostへ移す場合はこの式と座標変換を再確認してください。

4群は半開区間で重複なく240本を使い切ります。`beam` から観測indexへは19を加えます。

| pattern | beam index | 観測index |
|---|---|---|
| `lidar_front` | 0--29 と 210--239 | 19--48 と 229--258 |
| `lidar_left` | 150--209 | 169--228 |
| `lidar_rear` | 90--149 | 109--168 |
| `lidar_right` | 30--89 | 49--108 |

前方群が配列の先頭と末尾にまたがる点を、設定とテストで固定しています。未検出rayは1.0のまま、hitはBulletのhit fractionです。LiDAR距離や角度offsetを環境設定で変える場合、schemaの注記と実行時metadataへ実値を記録してください。240本を一行の「LiDAR」と省略しないため、`standard_259_schema()` は240行を生成します。

## provenance

標準行の参照元は次のclean checkoutです。配布versionだけで確認済みとは扱わず、commitとsource fileのsha256も保存します。

| source path | commit | sha256 |
|---|---|---|
| `metadrive/obs/state_obs.py` | `85e5dadc6c7436d324348f6e3d8f8e680c06b4db` | `2bf137e9f19388faa12069b93537b4d4f69cd1eca3d123c014637c8954c05c82` |
| `metadrive/component/navigation_module/node_network_navigation.py` | 同上 | `a423fdc5238d95aa4d1e01c6d614eff231d9e40fc24d40e7828c6874da05388f` |
| `metadrive/component/sensors/distance_detector.py` | 同上 | `ab12f0b9fd9ed9e5262d42c99294e007e20b361499767eee6dc527fb4eacfdf0` |
| `metadrive/utils/math.py` | 同上 | `0f96c6f59e9141cde0e6c3e4acfee8656a6062c9821056eceaf2d79a182679ef` |
| `metadrive/component/sensors/lidar.py` | 同上 | `a48e02d5ab49053b005fa4829db343af42e6e637f6c87e7ea84ce1cc7ef9683c` |
| `metadrive/component/map/base_map.py` | 同上 | `2f1b60cd529aa495e871de8a79490abbc26e7e032b9c7683a773bbb98682ec91` |
| `metadrive/component/vehicle/base_vehicle.py` | 同上 | `1660147eddb47faa8c2496bbef37ce369f6f82406a563cd34db7d9d9e4256f5a` |
| `metadrive/component/pg_space.py` | 同上 | `b87325023e7612913a24c447e8d4612f438595e7b8a357455db962fa4085134d` |
| `metadrive/base_class/base_object.py` | 同上 | `edfc48d63acdefafe5cff202374a6cf8ed06ffb327f01b55052352b0ddb74b50` |
| `metadrive/component/lane/straight_lane.py` | 同上 | `b27f2def9aee09fd2fa8465e7b51c9c549e2d40ff34e6892f898fa70d7b7552f` |
| `metadrive/component/lane/circular_lane.py` | 同上 | `45ddd24b4746313e4e5fd6d49548f041a2c891df8761624c8c185a5955c5be51` |

`source_identity()` exposes the same values programmatically, and each row carries the source path, import, line range, commit, dirty state, and applicable file hashes. Runtime adapters should additionally record their actual module paths and environment settings.

## 262移植テンプレートと合成schema

`template_262_schema()` は**262行すべて**を `host_input_unverified` として返します。259行を継ぎ足したり、末尾3行だけをunknownにしたりしません。移植先は各indexについて位置／順序、意味、正規化式、参照レーン、有効フラグを確認してから明示schemaを作ります。D=262だけではMetaDrive 259の意味を移しません。`load_schema(262)` または `input_schema_262_template.toml` はこの未確認テンプレートを選び、`require_verified=True` では拒否されます。

合成demoの `synthetic_259_contract` とT13用の `synthetic_262_contract` は、`SyntheticEnv` の人工的な幅だけを記述し、実MetaDrive／実移植先の確認済みschemaを意味しません。262幅で既定patternを自動生成せず、hostまたはsynthetic側のpatternを明示します。

schemaをCSVへ展開する例:

```python
from input_attribution.schema import standard_259_schema, write_schema_csv

write_schema_csv("outputs/input_attribution/schema_259.csv", standard_259_schema())
```
