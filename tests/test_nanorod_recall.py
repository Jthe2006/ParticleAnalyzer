import inspect
from unittest.mock import Mock

import numpy as np

from particleanalyzer.core.EnhancementPipeline import EnhancementPipeline
from particleanalyzer.core.ParticleAnalyzer import ParticleAnalyzer


def test_yolo_resolution_profile_is_not_hard_coded_to_640():
    assert ParticleAnalyzer._resolve_yolo_image_size("640x640") == 640
    assert ParticleAnalyzer._resolve_yolo_image_size("1024x1024") == 1024
    assert ParticleAnalyzer._resolve_yolo_image_size("1280x1280") == 1280
    assert ParticleAnalyzer._resolve_yolo_image_size("Original", (487, 554, 3)) == 576
    assert ParticleAnalyzer._resolve_yolo_image_size("Оригинал", (720, 480)) == 736


def test_yolo_receives_configured_inference_size(monkeypatch):
    model = Mock(return_value=[Mock(boxes=[])])
    analyzer = ParticleAnalyzer.__new__(ParticleAnalyzer)
    analyzer.device = "cpu"
    analyzer.default_lang = "en"
    analyzer.model_manager = Mock()
    analyzer.model_manager.get_model.return_value = model

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("gradio.Info", Mock())

    result = analyzer._process_with_yolo(
        model_change=ParticleAnalyzer.DEFAULT_NANOROD_MODEL,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        inference_size=1280,
        confidence_threshold=0.20,
        confidence_iou=0.50,
        number_detections=1000,
        pbar=Mock(),
        pr=Mock(),
    )

    assert result == (None, None, None)
    model.assert_called_once()
    call = model.call_args.kwargs
    assert call["imgsz"] == 1280
    assert call["conf"] == 0.20
    assert call["iou"] == 0.50
    assert call["max_det"] == 1000


def test_nanorod_recall_defaults_match_calibrated_preset():
    assert ParticleAnalyzer.DEFAULT_NANOROD_MODEL == "Yolo26 (dataset 11)"
    assert ParticleAnalyzer.DEFAULT_NANOROD_CONFIDENCE == 0.20
    assert ParticleAnalyzer.DEFAULT_NANOROD_IOU == 0.50
    assert ParticleAnalyzer.DEFAULT_NANOROD_SOLUTION == "1280x1280"
    assert ParticleAnalyzer.DEFAULT_NANOROD_ENHANCEMENT == "sem_edge_enhancement"


def test_every_enhancement_preset_executes_without_internal_errors(caplog):
    image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
    image = np.dstack((image, image, image))
    pipeline = EnhancementPipeline()

    for name in pipeline.get_available_pipelines():
        caplog.clear()
        enhanced = pipeline.apply_pipeline(image, name)

        assert enhanced.shape == image.shape
        assert enhanced.dtype == image.dtype
        assert "Ошибка на шаге" not in caplog.text


def test_overlap_option_is_appended_without_breaking_old_positional_calls():
    parameters = list(inspect.signature(ParticleAnalyzer.analyze_image).parameters)

    assert parameters[-3:] == [
        "pr",
        "selected_language",
        "exclude_overlapping_rods",
    ]
    assert (
        inspect.signature(ParticleAnalyzer.analyze_image)
        .parameters["exclude_overlapping_rods"]
        .default
        is False
    )
