## Layout

```
experiments/
├── _shared/
│   ├── sota_train_gpt.py   # decompressed bigbag SOTA (1.0810 BPB), reference
│   └── wandb_patch.py      # minimal wandb integration helpers
├── NN_short_name/
│   ├── train_gpt.py        # forked + modified copy of sota_train_gpt.py
│   ├── config.env          # env-var overrides for the run
│   └── notes.md            # what we're testing, why, expected gain
└── runs.md                 # cross-experiment log of results + WandB links
```

# Run an experiment:
cd experiments/NN_short_name
source config.env  # exports env vars
torchrun --standalone --nproc_per_node=3 train_gpt.py
```

`grad_accum_steps = 8 // world_size`, so `--nproc_per_node=3` keeps the same
786K-token effective batch, just at 6 micro-batches instead of 8.

## Reference SOTA env vars (bigbag, 1.0810)

```
QK_GAIN_INIT=5.25
TTT_ENABLED=1
TTT_LR=0.005
TTT_EPOCHS=3
LOOP_START=3 LOOP_END=5 NUM_LOOPS=2 ENABLE_LOOPING_AT=0.35
PARALLEL_RESIDUAL_START=7
MUON_WD=0.095 MATRIX_LR=0.022 EMA_DECAY=0.9965 WARMDOWN_FRAC=0.72
SEED=42
```

## WandB conventions

- Project: `parameter-golf` (override with `WANDB_PROJECT`)
- Run name: `RUN_ID` env var (set per experiment + seed, e.g. `01_pgsdclip_seed42`)
- Group: `WANDB_GROUP` for multi-seed runs of the same config
- Tags: include the experiment id and any lever-name tags