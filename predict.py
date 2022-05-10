# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from camtrapml.detection.models.megadetector import MegaDetectorV4_1
from camtrapml.detection.utils import render_detections
from camtrapml.image.utils import load_image

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.model = MegaDetectorV4_1()
        self.model.load_model()

    def predict(
        self,
        path: Path = Input(description="Input image"),
        min_score: float = 0.1,
    ) -> Path:
        """Run a single prediction on the model"""
        image = load_image(path)
        detections = self.model.detect(image, min_score=min_score)
        image = render_detections(path, detections, class_map=MegaDetectorV4_1.class_map)
        image.save('output.jpg')
        return Path('output.jpg')
