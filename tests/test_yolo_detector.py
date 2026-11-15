import tempfile
import unittest
from pathlib import Path

from yolo_detector import format_detection_line, parse_annotation_file, split_video_ids, yolo_labels


class AnnotationParsingTests(unittest.TestCase):
    def test_rows_align_by_frame_number_when_a_frame_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.txt"
            path.write_text(
                "time_layer: 1 detections: (10, 20, 30, 40), \n"
                "time_layer: 3 detections: \n",
                encoding="utf-8",
            )
            self.assertEqual(
                parse_annotation_file(path),
                {1: [(10, 20, 30, 40)], 3: []},
            )

    def test_rejects_malformed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.txt"
            path.write_text("not an annotation\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "malformed annotation row"):
                parse_annotation_file(path)


class DatasetPreparationTests(unittest.TestCase):
    def test_splits_are_disjoint_and_hold_video_14_out(self):
        video_ids = range(1, 51)
        splits = split_video_ids(video_ids, seed=42)
        self.assertIn(14, splits["test"])
        self.assertEqual((len(splits["train"]), len(splits["val"]), len(splits["test"])), (35, 7, 8))
        self.assertEqual(
            set(splits["train"] + splits["val"] + splits["test"]), set(video_ids)
        )
        self.assertFalse(set(splits["train"]) & set(splits["val"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["val"]) & set(splits["test"]))

    def test_converts_boxes_to_normalized_yolo_labels(self):
        self.assertEqual(
            yolo_labels([(10, 20, 30, 60)], width=100, height=100),
            ["0 0.400000 0.200000 0.400000 0.200000"],
        )

    def test_prediction_line_uses_baseline_frame_and_box_order(self):
        self.assertEqual(
            format_detection_line(3, [(20.0, 10.0, 40.0, 30.0)], 100, 100),
            "time_layer: 3 detections: (10, 20, 30, 40), \n",
        )

    def test_rejects_boxes_without_area(self):
        with self.assertRaisesRegex(ValueError, "Invalid or out-of-frame box"):
            yolo_labels([(10, 20, 10, 20)], width=100, height=100)


if __name__ == "__main__":
    unittest.main()
