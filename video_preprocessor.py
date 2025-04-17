import cv2
import numpy as np
from pathlib import Path


class VideoPreprocessor:
    def __init__(self, normalize_brightness=True, crop=True):
        self.enable_normalization = normalize_brightness
        self.enable_crop = crop

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype='float32')
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def is_within_frame(self, region, shape):
        h, w = shape[:2]
        return np.all(
            (region[:, 0] >= 0) &
            (region[:, 0] < w) &
            (region[:, 1] >= 0) &
            (region[:, 1] < h)
        )

    def find_region(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.inRange(hsv, (0, 0, 120), (180, 255, 255))
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 60))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        bright_dilated = cv2.dilate(bright_mask, kernel, iterations=1)
        surrounded_dark = cv2.bitwise_and(dark_mask, bright_dilated)

        cnts, _ = cv2.findContours(surrounded_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area < frame.shape[0] * frame.shape[1] * 0.2:
            return None

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            return self.order_points(approx.reshape(4, 2))
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            return self.order_points(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype="float32"))

    def extract_anchor_region(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        # fallback value
        sizes = [(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))]
        norm_regions = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 10 != 0:
                frame_idx += 1
                continue

            region = self.find_region(frame)
            if region is not None and self.is_within_frame(region, frame.shape):
                h, w = frame.shape[:2]
                norm_region = region / np.array([[w, h]])
                norm_regions.append(norm_region)

                width = int(max(np.linalg.norm(region[0] - region[1]), np.linalg.norm(region[2] - region[3])))
                height = int(max(np.linalg.norm(region[0] - region[3]), np.linalg.norm(region[1] - region[2])))
                sizes.append((width, height))

            frame_idx += 1

        cap.release()

        if len(norm_regions) < 5:
            raise Exception("Not enough valid region samples found.")

        anchor = np.mean(norm_regions, axis=0)
        mean_w = int(np.mean([s[0] for s in sizes]))
        mean_h = int(np.mean([s[1] for s in sizes]))

        return anchor, mean_w, mean_h

    def normalize_brightness(self, frame, target_brightness=130, strength=0.5):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)

        current_mean = np.mean(v)
        ratio = target_brightness / (current_mean + 1e-6)
        ratio = max(1 - strength, min(1 + strength, ratio))

        v *= ratio
        v = np.clip(v, 0, 255)
        hsv[:, :, 2] = v.astype(np.uint8)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process_video(self, video_path, output_path):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        mean_w, mean_h = w, h
        anchor_px = None

        if self.enable_crop:
            try:
                anchor_norm, mean_w, mean_h = self.extract_anchor_region(video_path)
                anchor_px = (anchor_norm * np.array([[w, h]], dtype=np.float32)).astype(np.float32)
                dst = np.array([[0, 0], [mean_w - 1, 0], [mean_w - 1, mean_h - 1], [0, mean_h - 1]], dtype=np.float32)
            except Exception as e:
                print(f"[{video_path.name}] cropping disabled due to error: {e}")
                anchor_px = None 

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (mean_w, mean_h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.enable_normalization:
                frame = self.normalize_brightness(frame)

            if self.enable_crop and anchor_px is not None:
                M = cv2.getPerspectiveTransform(anchor_px, dst)
                frame = cv2.warpPerspective(frame, M, (mean_w, mean_h))
            else:
                frame = cv2.resize(frame, (mean_w, mean_h))

            out.write(frame)

        cap.release()
        out.release()

    def process_directory(self, input_dir="videos", output_dir="videos/output"):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for video_path in input_dir.glob("*.mp4"):
            print(f"Processing: {video_path.name}")
            output_path = output_dir / video_path.name
            self.process_video(video_path, output_path)


if __name__ == "__main__":
    preprocessor = VideoPreprocessor(normalize_brightness=True, crop=True)
    preprocessor.process_directory("videos", "videos/output")
