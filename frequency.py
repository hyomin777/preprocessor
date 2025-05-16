import cv2
import numpy as np
import torch

import torch
import numpy as np
from PIL import Image


def temporal_highpass(pil_list, keep_ratio=0.4):
    arr = np.stack([np.array(img) for img in pil_list], axis=0)
    T, H, W, C = arr.shape

    x = torch.from_numpy(arr).float().permute(3, 0, 1, 2)  # (C, T, H, W)

    freq = torch.fft.fft(x, dim=1)
    freq_shifted = torch.fft.fftshift(freq, dim=1)

    ft = torch.fft.fftfreq(T).abs()  # (T,)
    mask = (ft >= keep_ratio)  # high freq True
    mask = mask[None, :, None, None]  # (1, T, 1, 1) broadcasting

    freq_filtered = freq_shifted * mask
    freq_unshifted = torch.fft.ifftshift(freq_filtered, dim=1)
    x_out = torch.fft.ifft(freq_unshifted, dim=1).real  # (C, T, H, W)

    x_out = x_out.permute(1, 2, 3, 0).contiguous()  # (T, H, W, C)
    x_min = x_out.amin(dim=(1, 2), keepdim=True)
    x_max = x_out.amax(dim=(1, 2), keepdim=True)
    x_out = (x_out - x_min) / (x_max - x_min + 1e-6) * 255
    x_out = x_out.clamp(0, 255).byte().cpu().numpy()  # (T, H, W, C)

    pil_out = [Image.fromarray(x_out[t]) for t in range(T)]
    return pil_out


def make_clip(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (256, 256))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        frames.append(frame_pil)
    cap.release()
    return frames, fps


def save_video_file(frames, fps):
    frame_np = np.array(frames[0])
    height, width = frame_np.shape[:2]

    out = cv2.VideoWriter(
        'ouput.mp4',
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    for img in frames:
        frame_np = np.array(img)
        if frame_np.shape[2] == 3:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_np)

    out.release()


def main(video_path):
    frames, fps = make_clip(video_path)
    frames = temporal_highpass(frames)
    save_video_file(frames, fps)


if __name__ == '__main__':
    video_path = 'sample.mp4'

    main(video_path)
