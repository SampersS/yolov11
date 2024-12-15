#https://docs.openvino.ai/2024/notebooks/yolov11-quantization-with-accuracy-control-with-output.html

import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.validator import BaseValidator as Validator
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix

ROOT = os.path.abspath("")

MODEL_NAME = "yolo11n"

model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")
args = get_cfg(cfg=DEFAULT_CFG)
args.data = "coco128.yaml"

import openvino as ov
model_path = Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
if not model_path.exists():
    model.export(format="openvino", dynamic=True, half=False)

ov_model = ov.Core().read_model(model_path)

from ultralytics.data.converter import coco80_to_coco91_class


validator = model.task_map[model.task]["validator"](args=args)
validator.data = check_det_dataset(args.data)
validator.stride = 3

data_loader = validator.get_dataloader("./testMaterie/COCOimages",1)

validator.is_coco = True
validator.class_map = coco80_to_coco91_class()
validator.names = model.model.names
validator.metrics.names = validator.names
validator.nc = model.model.model[-1].nc
validator.nm = 32
validator.process = ops.process_mask
validator.plot_masks = []

