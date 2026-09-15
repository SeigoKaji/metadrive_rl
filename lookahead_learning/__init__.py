"""既存ホストへ組み込む前方注視 runtime。

運用時はホスト root の通常入口へ同じTOMLを渡します::

    python train.py --config configs/your_experiment.toml
    python evaluate.py --config configs/your_experiment.toml

config loader は任意の ``[lookahead]`` table を
``lookahead_learning.checkpoint.resolve_lookahead_config(raw.get("lookahead"))``
で解決し、共通env factoryは設定がある場合だけ
``lookahead_learning.adapter.wrap_lookahead_env(raw_env, **config)``
で既存Envを包みます。raw観測が1次元float32の幅Dなら、tableありの観測は
既存prefixを保ったままD+3です。tableなしはbaseline、``pp_weight=0.0``は
PP項なし、正の値はPP不一致ペナルティも追加します。新しい
``lateral_accel_reward_enabled``はPPとは独立に参照経路の必要横加速度の
超過ペナルティを切り替えます。省略時Offで、観測は同じD+3です。

既存ホストの観測生成、reward_function、終了条件、Action適用、車両とNavigationの
意味はホストとadapterの責務です。PPO保存前に
``set_lookahead_model_metadata``、ロード後に
``validate_lookahead_model_metadata``を呼び、設定とschema version 2をZIP内属性で
照合します。schema v1の旧モデルは新項Offとして読み取れます。
詳細は [追加報酬](docs/lateral_acceleration_reward.md)、 [docs/README](docs/README.md)、
[移植手順](docs/porting.md)、[仕組みと数式](docs/methods.md) を参照してください。
"""

__version__ = "0.2.0"
