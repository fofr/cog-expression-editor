import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "face_yolov8n.pt",
                "appearance_feature_extractor.safetensors",
                "motion_extractor.safetensors",
                "spade_generator.safetensors",
                "stitching_retargeting_module.safetensors",
                "warping_module.safetensors",
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def update_workflow(self, workflow, **kwargs):
        workflow["15"]["inputs"]["image"] = kwargs["image_filename"]

        expression_editor = workflow["14"]["inputs"]
        expression_editor["rotate_pitch"] = kwargs["rotate_pitch"]
        expression_editor["rotate_yaw"] = kwargs["rotate_yaw"]
        expression_editor["rotate_roll"] = kwargs["rotate_roll"]
        expression_editor["blink"] = kwargs["blink"]
        expression_editor["eyebrow"] = kwargs["eyebrow"]
        expression_editor["wink"] = kwargs["wink"]
        expression_editor["pupil_x"] = kwargs["pupil_x"]
        expression_editor["pupil_y"] = kwargs["pupil_y"]
        expression_editor["aaa"] = kwargs["aaa"]
        expression_editor["eee"] = kwargs["eee"]
        expression_editor["woo"] = kwargs["woo"]
        expression_editor["smile"] = kwargs["smile"]
        expression_editor["src_ratio"] = kwargs["src_ratio"]
        expression_editor["sample_ratio"] = kwargs["sample_ratio"]
        expression_editor["crop_factor"] = kwargs["crop_factor"]

    def predict(
        self,
        image: Path = Input(
            description="Image of a face",
            default=None,
        ),
        rotate_pitch: float = Input(
            default=0,
            ge=-20,
            le=20,
            description="Rotation pitch: Adjusts the up and down tilt of the face",
        ),
        rotate_yaw: float = Input(
            default=0,
            ge=-20,
            le=20,
            description="Rotation yaw: Adjusts the left and right turn of the face",
        ),
        rotate_roll: float = Input(
            default=0,
            ge=-20,
            le=20,
            description="Rotation roll: Adjusts the tilt of the face to the left or right",
        ),
        blink: float = Input(
            default=0,
            ge=-20,
            le=5,
            description="Blink: Controls the degree of eye closure",
        ),
        eyebrow: float = Input(
            default=0,
            ge=-10,
            le=15,
            description="Eyebrow: Adjusts the height and shape of the eyebrows",
        ),
        wink: float = Input(
            default=0,
            ge=0,
            le=25,
            description="Wink: Controls the degree of one eye closing",
        ),
        pupil_x: float = Input(
            default=0,
            ge=-15,
            le=15,
            description="Pupil X: Adjusts the horizontal position of the pupils",
        ),
        pupil_y: float = Input(
            default=0,
            ge=-15,
            le=15,
            description="Pupil Y: Adjusts the vertical position of the pupils",
        ),
        aaa: float = Input(
            default=0,
            ge=-30,
            le=120,
            description="AAA: Controls the mouth opening for 'aaa' sound",
        ),
        eee: float = Input(
            default=0,
            ge=-20,
            le=15,
            description="EEE: Controls the mouth shape for 'eee' sound",
        ),
        woo: float = Input(
            default=0,
            ge=-20,
            le=15,
            description="WOO: Controls the mouth shape for 'woo' sound",
        ),
        smile: float = Input(
            default=0,
            ge=-0.3,
            le=1.3,
            description="Smile: Adjusts the degree of smiling",
        ),
        src_ratio: float = Input(default=1, ge=0, le=1, description="Source ratio"),
        sample_ratio: float = Input(
            default=1, ge=-0.2, le=1.2, description="Sample ratio"
        ),
        crop_factor: float = Input(
            default=1.7, ge=1.5, le=2.5, description="Crop factor"
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        self.comfyUI.connect()

        image_filename = None
        if image:
            image_filename = self.filename_with_extension(image, "image")
            self.handle_input_file(image, image_filename)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            image_filename=image_filename,
            rotate_pitch=rotate_pitch,
            rotate_yaw=rotate_yaw,
            rotate_roll=rotate_roll,
            blink=blink,
            eyebrow=eyebrow,
            wink=wink,
            pupil_x=pupil_x,
            pupil_y=pupil_y,
            aaa=aaa,
            eee=eee,
            woo=woo,
            smile=smile,
            src_ratio=src_ratio,
            sample_ratio=sample_ratio,
            crop_factor=crop_factor,
        )

        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(COMFYUI_TEMP_OUTPUT_DIR)
        )
