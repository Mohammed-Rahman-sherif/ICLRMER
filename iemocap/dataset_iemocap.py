import pickle
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import config_iemocap as ie_cfg
from graph_utils import build_hetero_graph_for_dialogue

class IEMOCAPHeteroDataset(Dataset):
    def __init__(self, path, dataset, split='train', split_ratio=0.1, random_seed=33, text_feature_key='videoText1'):
        self.dataset = dataset
        if self.dataset == 'iemocap':
            self.cfg = ie_cfg
            self.cfg.TEXT_FEATURE_KEY = text_feature_key
            data = pickle.load(open(path, "rb"), encoding="latin1")
            (self.videoIDs, self.videoSpeakers, self.videoLabels,
             self.videoText1, self.videoText2, self.videoText3, self.videoText4,
             self.videoAudio, self.videoVisual, self.videoTranscripts,
             self.trainVid, self.testVid) = data
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        all_speakers = [speaker for vid_id in self.videoIDs for speaker in self.videoSpeakers[vid_id]]
        unique_speakers = sorted(list(set(all_speakers)))
        self.speaker_to_idx = {speaker: i for i, speaker in enumerate(unique_speakers)}
        self.num_speakers = len(unique_speakers)
        
        train_keys, val_keys = train_test_split(list(self.trainVid), test_size=split_ratio, random_state=random_seed)

        if split == 'train':
            self.keys = train_keys
        elif split == 'val':
            self.keys = val_keys
        elif split == 'test':
            self.keys = list(self.testVid)
        else:
            raise ValueError(f"Invalid split name: {split}. Choose from 'train', 'val', 'test'.")

        self.labels = []
        for vid in self.keys:
            self.labels.extend(self.videoLabels[vid])

        num_dialogues = len(self.keys)
        num_utterances = len(self.labels)
        print(f"{split}-{self.dataset.upper()}: {num_utterances} utterances from {num_dialogues} dialogues")

    def __len__(self):
        return len(self.keys)

    def _pick_text_matrix(self, vid):
        return getattr(self, self.cfg.TEXT_FEATURE_KEY)[vid]

    def __getitem__(self, idx):
        vid = self.keys[idx]
        txt = torch.as_tensor(np.array(self._pick_text_matrix(vid)), dtype=torch.float32)
        aud = torch.as_tensor(np.array(self.videoAudio[vid]), dtype=torch.float32)
        vis = torch.as_tensor(np.array(self.videoVisual[vid]), dtype=torch.float32)
        labels = torch.as_tensor(self.videoLabels[vid], dtype=torch.long)
        
        speakers = self.videoSpeakers[vid]
        speaker_indices = torch.tensor([self.speaker_to_idx[s] for s in speakers], dtype=torch.long)
        
        utterance_ids = self.videoIDs[vid]
        utterance_texts = self.videoTranscripts[vid]

        # Assuming your build_hetero_graph_for_dialogue does not need speaker_indices
        graph = build_hetero_graph_for_dialogue(
                txt, aud, vis, labels, speakers, vid,
                node_types=self.cfg.NODE_TYPES
        )
        
        graph.speaker_idx = speaker_indices
        graph.utterance_ids = utterance_ids
        graph.utterance_texts = utterance_texts
        return graph