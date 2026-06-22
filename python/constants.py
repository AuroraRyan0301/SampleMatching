import os
from os.path import realpath, join

REPO_DIR = realpath(join(os.path.dirname(__file__), '..'))

# Root of the downloadable data archive (scenes, reference images, GT volumes).
# Download the data archive and unzip it here, or point POSTTRACKING_DATA_DIR
# elsewhere. Expected layout:
#   $DATA_DIR/scenes/          scene XMLs + .vol grids + textures (per-scene dirs)
#   $DATA_DIR/mi_ref/<scene>/  multi-view reference images (ref_XXXXXX.exr)
DATA_DIR = realpath(os.environ.get('POSTTRACKING_DATA_DIR', join(REPO_DIR, 'data')))
SCENE_DIR = join(DATA_DIR, 'scenes')
REF_DIR = join(DATA_DIR, 'mi_ref')

# Override via env var POSTTRACKING_OUTPUT_DIR to send experiment outputs to a
# different location (e.g. a dedicated reproduction run) without touching code.
OUTPUT_DIR = realpath(os.environ.get(
    'POSTTRACKING_OUTPUT_DIR', join(REPO_DIR, 'output')))

# Override via env var POSTTRACKING_WANDB_PROJECT to log to a different project.
WANDB_EXP_NAME = os.environ.get('POSTTRACKING_WANDB_PROJECT', 'delta-formal')