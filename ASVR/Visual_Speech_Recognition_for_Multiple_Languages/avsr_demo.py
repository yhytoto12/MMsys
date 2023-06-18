import torch, torchaudio, torchvision
import hydra
from pipelines.pipeline import InferencePipeline
from viztracer import log_sparse
import torch.profiler



class VideoStream():
    """ Dummy video stream for testing purposes. Reads a file using torchvision and can then be read from later. """
    def __init__(self, filepath):
        self.frames = torchvision.io.read_video(filepath, pts_unit='sec')[0].numpy()
    def read(self):
        return self.frames

class AudioStream():
    """ Dummy audio stream for testing purposes. Reads a file using torchaudio and can then be read from later. """
    def __init__(self, filepath):
        self.waveform, self.sample_rate = torchaudio.load(filepath, normalize=True)
    def read(self):
        return self.waveform, self.sample_rate

@hydra.main(version_base=None, config_path="hydra_configs", config_name="default")
@log_sparse(stack_depth=10)
def main(cfg):
    video_stream =  VideoStream(cfg.data_filename+'.mp4')
    audio_stream = AudioStream(cfg.data_filename+'.wav')

    device = torch.device(f"cuda:{cfg.gpu_idx}" if torch.cuda.is_available() and cfg.gpu_idx >= 0 else "cpu")
    print(f"device: {device}")

    output = InferencePipeline(cfg.config_filename, device=device, detector=cfg.detector, face_track=True)(video_stream,audio_stream, cfg.landmarks_filename, cfg.parallel)
    print(f"hyp: {output}")


if __name__ == '__main__':
    main()
