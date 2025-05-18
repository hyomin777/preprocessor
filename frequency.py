import random
import cv2
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


class FrequencyAug:
    def __init__(self, keep_ratio=0.1, filter_type='high'):
        self.keep_ratio = keep_ratio
        self.filter_type = filter_type
        assert filter_type in ['low', 'high'], "filter_type must be 'low' or 'high'"

    def __call__(self, clip):
        frames = [np.array(f.convert("RGB")) for f in clip]
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        print(f'Input video shape: {video.shape}')
        
        return self._filter_frames(video)
    
    def _filter_frames(self, video):
        T, H, W, C = video.shape
        
        # C, T, H, W
        video_cthw = np.transpose(video, (3, 0, 1, 2))
        video_tensor = torch.tensor(video_cthw, dtype=torch.float32)
        
        # 3D FFT (T, H, W)
        freq = torch.fft.fftn(video_tensor, dim=(1, 2, 3))
        freq_shifted = torch.fft.fftshift(freq, dim=(1, 2, 3))
        
        # Frequncy mask
        mask = self._create_frequency_mask(T, H, W, C)
        
        if self.filter_type == 'low':
            filtered = freq_shifted * mask
        else:
            filtered = freq_shifted * (~mask)
        
        filtered_shifted = torch.fft.ifftshift(filtered, dim=(1, 2, 3))
        img_filtered = torch.fft.ifftn(filtered_shifted, dim=(1, 2, 3)).real
        
        img_filtered = self._normalize_tensor(img_filtered)
        
        # (C, T, H, W) → (T, H, W, C)
        img_out = img_filtered.permute(1, 2, 3, 0).numpy()
        print(f'Output image shape: {img_out.shape}')
        
        return [Image.fromarray(img_out[t].astype(np.uint8)) for t in range(T)]
    
    def _create_frequency_mask(self, T, H, W, C):
        center_t = T // 2
        center_h = H // 2
        center_w = W // 2
        
        t_coords = torch.arange(T) - center_t
        h_coords = torch.arange(H) - center_h
        w_coords = torch.arange(W) - center_w
        
        t_grid, h_grid, w_grid = torch.meshgrid(t_coords, h_coords, w_coords, indexing='ij')

        distance = torch.sqrt(
            (t_grid / (T/2))**2 + 
            (h_grid / (H/2))**2 + 
            (w_grid / (W/2))**2
        )
        
        threshold = self.keep_ratio
        
        mask_3d = distance <= threshold
        
        mask = mask_3d.unsqueeze(0).expand(C, -1, -1, -1)
        
        return mask

    def _normalize_tensor(self, tensor):
        min_val = tensor.min()
        tensor = tensor - min_val
        
        max_val = tensor.max()
        if max_val > 0:
            tensor = 255 * (tensor / max_val)
        
        return tensor.clamp(0, 255).byte()
    
class StochasticFrequencyAug:
    def __init__(
        self,
        spatial_mask_ratio=0.1,
        temporal_mask_ratio=0.1,
        spatial_prob=1.0,  
        temporal_prob=1.0, 
        filter_type="high",
        mask_type="band",
        stochastic_strength=(0.0, 1.0)
    ):
        self.spatial_mask_ratio = spatial_mask_ratio
        self.temporal_mask_ratio = temporal_mask_ratio
        self.spatial_prob = spatial_prob
        self.temporal_prob = temporal_prob
        
        assert filter_type in ["low", "high"], "filter_type must be 'low' or 'high'"
        self.filter_type = filter_type
        
        assert mask_type in ["band", "block"], "mask_type must be 'band' or 'block'"
        self.mask_type = mask_type
        
        self.stochastic_strength = stochastic_strength

    def __call__(self, clip):
        frames = [np.array(f.convert("RGB")) for f in clip]
        video = np.stack(frames, axis=0)  # (T, H, W, C)
        print(f"Input video shape: {video.shape}")

        filtered_video = self._filter_video(video)
        return [Image.fromarray(filtered_video[t].astype(np.uint8)) for t in range(filtered_video.shape[0])]

    def _filter_video(self, video):
        T, H, W, C = video.shape
        filtered_video = np.zeros_like(video, dtype=np.float32)

        for c in range(C):
            # T, H, W
            video_c = video[..., c].astype(np.float32)
            
            # Temporal Filtering
            if random.random() < self.temporal_prob:
                video_c = self._apply_temporal_filtering(video_c)
            
            # Spatial Filtering
            if random.random() < self.spatial_prob:
                video_c = self._apply_spatial_filtering(video_c)
            
            filtered_video[..., c] = video_c

        return self._normalize_video(filtered_video)

    def _apply_temporal_filtering(self, video_c):
        T, H, W = video_c.shape
        filtered = np.zeros_like(video_c)
        
        for h in range(H):
            for w in range(W):
                temporal_signal = video_c[:, h, w]
                
                freq = torch.fft.fft(torch.tensor(temporal_signal))
                freq_shifted = torch.fft.fftshift(freq)
                
                mask = self._create_1d_mask(T)
                filtered_freq = freq_shifted * mask
                
                filtered_signal = torch.fft.ifftshift(filtered_freq)
                filtered_signal = torch.fft.ifft(filtered_signal).real.numpy()

                filtered[:, h, w] = filtered_signal
        
        return filtered

    def _apply_spatial_filtering(self, video_c):
        T, H, W = video_c.shape
        filtered = np.zeros_like(video_c)
        
        strength = random.uniform(*self.stochastic_strength)

        for t in range(T):
            # (H, W)
            frame = video_c[t]
            
            # 2D-FFT
            freq = torch.fft.fft2(torch.tensor(frame))
            freq_shifted = torch.fft.fftshift(freq)
            
            mask = self._create_2d_mask(H, W, strength)
            filtered_freq = freq_shifted * mask
            
            filtered_frame = torch.fft.ifftshift(filtered_freq)
            filtered_frame = torch.fft.ifft2(filtered_frame).real.numpy()
            
            filtered[t] = filtered_frame
        
        return filtered

    def _create_1d_mask(self, size):
        center = size // 2
        coords = torch.arange(size) - center
        
        # Normalized distance
        distance = torch.abs(coords / (size/2))
        
        # Stochastic Strength
        strength = random.uniform(*self.stochastic_strength)
        threshold = self.temporal_mask_ratio * strength
        
        if self.mask_type == "band":
            if self.filter_type == "low":
                mask = distance <= threshold
            else:
                mask = distance > threshold
        else:
            mask = torch.ones(size, dtype=torch.bool)
            num_blocks = int(size * threshold)
            if num_blocks > 0:
                block_indices = torch.randperm(size)[:num_blocks]
                mask[block_indices] = False
                if self.filter_type == "high":
                    mask = ~mask
        
        return mask

    def _create_2d_mask(self, height, width, strength=1.0):
        center_h = height // 2
        center_w = width // 2
        
        h_coords = torch.arange(height) - center_h
        w_coords = torch.arange(width) - center_w
        h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing='ij')
        
        distance = torch.sqrt(
            (h_grid / (height/2))**2 + 
            (w_grid / (width/2))**2
        )
        
        threshold = self.spatial_mask_ratio * strength
        
        if self.mask_type == "band":
            if self.filter_type == "low":
                mask = distance <= threshold
            else:
                mask = distance > threshold
        else:
            mask = torch.ones((height, width), dtype=torch.bool)
            num_blocks = int(height * width * threshold)
            if num_blocks > 0:
                indices = torch.randperm(height * width)[:num_blocks]
                h_indices = indices // width
                w_indices = indices % width
                mask[h_indices, w_indices] = False
                if self.filter_type == "high":
                    mask = ~mask
        
        return mask

    def _normalize_video(self, video):
        result = np.zeros_like(video)
        
        for c in range(video.shape[-1]):
            channel = video[..., c]
            min_val = channel.min()
            max_val = channel.max()
            
            if max_val > min_val:
                normalized = 255 * (channel - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(channel)
            
            result[..., c] = normalized
        
        return np.clip(result, 0, 255).astype(np.uint8)


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
    freq_aug = StochasticFrequencyAug()
    frames, fps = make_clip(video_path)
    frames = freq_aug(frames)
    save_video_file(frames, fps)


if __name__ == '__main__':
    video_path = 'sample.mp4'

    main(video_path)
