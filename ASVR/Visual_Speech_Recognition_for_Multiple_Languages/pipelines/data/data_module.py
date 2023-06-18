#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch
import torchaudio
import torchvision
from .transforms import AudioTransform, VideoTransform
from torch.profiler import record_function
# from multiprocessing.pool import ThreadPool
import concurrent.futures
import pdb



class AVSRDataLoader:
    def __init__(self, modality, speed_rate=1, transform=True, detector="retinaface", convert_gray=True):
        self.modality = modality
        self.transform = transform
        if self.modality in ["audio", "audiovisual"]:
            self.audio_transform = AudioTransform()
        if self.modality in ["video", "audiovisual"]:
            if detector == "mediapipe":
                from pipelines.detectors.mediapipe.video_process import VideoProcess
                self.video_process = VideoProcess(convert_gray=convert_gray)
                from pipelines.detectors.mediapipe.detector import LandmarksDetector
                self.landmarks_detector = LandmarksDetector()
            if detector == "retinaface":
                from pipelines.detectors.retinaface.video_process import VideoProcess
                self.video_process = VideoProcess(convert_gray=convert_gray)
            self.video_transform = VideoTransform(speed_rate=speed_rate)


    def load_data(self, video_stream, audio_stream, landmarks=None, transform=True,parallel=False):
    # def load_data(self, data_filename, landmarks=None, transform=True):
        if self.modality == "audio":
            audio, sample_rate = self.load_audio(data_filename)
            audio = self.audio_process(audio, sample_rate)
            return self.audio_transform(audio) if self.transform else audio
        if self.modality == "video":
            video = self.load_video(data_filename)
            video = self.video_process(video, landmarks)
            video = torch.tensor(video)
            return self.video_transform(video) if self.transform else video
        if self.modality == "audiovisual":
            
            rate_ratio = 640
            
            # audio, sample_rate = self.load_audio(data_filename)
            # pool = Pool(processes=2)

             # tuple of args for foo

            # do some other stuff in the main process

            
            if parallel==False:
            #     # audio_thread = threading.Thread(target=self.process_audio(audio_stream))
            # with record_function("LOADING AUDIO STREAM"):
            #     async_result = pool.apply_async(self.process_audio, (audio_stream,))
                with record_function("LOADING AUDIO STREAM"):
                    audio, sample_rate = self.load_audio(audio_stream)
                with record_function("PROCESSING AUDIO"):
                    audio = self.audio_process(audio, sample_rate)
                
                with record_function("LOADING VIDEO STREAM"):
                    video = self.load_video(video_stream)
                with record_function("PROCESSING VIDEO"):
                    video = self.video_process(video, landmarks)
                video = torch.tensor(video)
                min_t = min(len(video), audio.size(1) // rate_ratio)
                # pdb.set_trace()
                audio = audio[:, :min_t*rate_ratio]
                video = video[:min_t]
                with record_function("TRANSFORMS"):
                    if self.transform:
                        audio = self.audio_transform(audio)
                        video = self.video_transform(video)
                return video, audio
            if parallel==True:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    with record_function("AUDIO THREAD"):
                        audio_future = executor.submit(self.process_audio,audio_stream)
                        # async_result = pool.apply_async(self.process_audio, (audio_stream,))
                    with record_function("VIDEO THREAD"):
                        video_future = executor.submit(self.process_video,video_stream)
                    # test = executor.submit(self.test)
                        # async_result2 = pool.apply_async(self.process_video, (video_stream,landmarks))
                    audio=audio_future.result()
                    video=video_future.result()
                return video, audio
            
            # 149
            

    # def test():


    # def load_audio(self, data_filename):
    #     # waveform, sample_rate = audio_stream.read()
    #     waveform, sample_rate = torchaudio.load(data_filename, normalize=True)
    #     return waveform, sample_rate


    # def load_video(self, data_filename):
    #     # return video_stream.read()
    #     return torchvision.io.read_video(data_filename, pts_unit='sec')[0].numpy()

    def process_audio(self, audio_stream):
        audio, sample_rate = self.load_audio(audio_stream)
        audio = self.audio_process(audio, sample_rate)
        min_t=149
        rate_ratio = 640
        audio = audio[:, :min_t*rate_ratio]
        if self.transform:
            audio = self.audio_transform(audio)
        return audio

    def process_video(self, video_stream):
        # get landmarks first
        video = self.load_video(video_stream)
        landmarks = self.landmarks_detector(video_stream)
        video = self.video_process(video, landmarks)
        video = torch.tensor(video)
        min_t=149
        video = video[:min_t]
        if self.transform:
            video = self.video_transform(video)
        return video

    def load_audio(self, audio_stream):
        waveform, sample_rate = audio_stream.read()
        return waveform, sample_rate

    def load_video(self, video_stream):
        return video_stream.read()


    def audio_process(self, waveform, sample_rate, target_sample_rate=16000):
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
