# # graph_utils.py (the recommended new version)

# from __future__ import annotations
# import torch
# from torch_geometric.data import HeteroData

# def build_hetero_graph_for_dialogue(
#         text_f, audio_f, visual_f, labels, speakers, vid, # <-- Added 'speakers'
#         *, node_types):
#     """
#     Builds a single graph for an entire dialogue.
#     Nodes represent utterances.
#     This version uses speaker-aware temporal edges.
#     """
#     data = HeteroData()
#     feats = dict(text=text_f, audio=audio_f, visual=visual_f)

#     for nt, feat in feats.items():
#         data[nt].x = feat

#     num_utterances = next(iter(data.x_dict.values())).shape[0]

#     # --- NEW: Speaker-aware temporal edges (DialogueGCN style) ---
#     # This creates unidirectional edges based on speaker turn dynamics.
    
#     # Edges for utterances from the SAME speaker
#     same_speaker_src, same_speaker_dst = [], []
#     # Edges for utterances from a DIFFERENT speaker
#     other_speaker_src, other_speaker_dst = [], []

#     # For each utterance 'j' (destination)...
#     for j in range(num_utterances):
#         # ...look at all previous utterances 'i' (source).
#         for i in range(j):
#             # If the speaker is the same, add a 'past_same' edge
#             if speakers[i] == speakers[j]:
#                 same_speaker_src.append(i)
#                 same_speaker_dst.append(j)
#             # If the speaker is different, add a 'past_other' edge
#             else:
#                 other_speaker_src.append(i)
#                 other_speaker_dst.append(j)
    
#     # Add these new edge types to the graph for each modality
#     for nt in data.node_types:
#         # Add 'past_same' edges
#         edge_index_same = torch.tensor([same_speaker_src, same_speaker_dst], dtype=torch.long)
#         data[nt, 'past_same', nt].edge_index = edge_index_same
        
#         # Add 'past_other' edges
#         edge_index_other = torch.tensor([other_speaker_src, other_speaker_dst], dtype=torch.long)
#         data[nt, 'past_other', nt].edge_index = edge_index_other

#     # --- Cross-modal edges (unconditional) ---
#     # This section remains unchanged.
#     nodes = torch.arange(num_utterances)
#     edge_index = torch.stack([nodes, nodes], 0)
#     pairs = [("text", "audio"), ("text", "visual"), ("audio", "visual")]
#     for mod_a, mod_b in pairs:
#         data[mod_a, f"{mod_a}_to_{mod_b}", mod_b].edge_index = edge_index
#         data[mod_b, f"{mod_b}_to_{mod_a}", mod_a].edge_index = edge_index

#     data.y = labels
#     data.vid = vid
#     return data




from __future__ import annotations
import torch
from torch_geometric.data import HeteroData

def build_hetero_graph_for_dialogue(
        text_f, audio_f, visual_f, labels, speakers, vid,
        *, node_types):
    """
    Builds a single graph for an entire dialogue.
    Nodes represent utterances.
    This version uses speaker-aware temporal edges.
    """
    data = HeteroData()
    feats = dict(text=text_f, audio=audio_f, visual=visual_f)

    for nt, feat in feats.items():
        data[nt].x = feat

    num_utterances = next(iter(data.x_dict.values())).shape[0]

    # --- Speaker-aware temporal edges (DialogueGCN style) ---
    # This creates unidirectional edges based on speaker turn dynamics.
    
    # Edges for utterances from the SAME speaker
    same_speaker_src, same_speaker_dst = [], []
    # Edges for utterances from a DIFFERENT speaker
    other_speaker_src, other_speaker_dst = [], []

    # For each utterance 'j' (destination)...
    for j in range(num_utterances):
        # ...look at all previous utterances 'i' (source).
        for i in range(j):
            # If the speaker is the same, add a 'past_same' edge
            if speakers[i] == speakers[j]:
                same_speaker_src.append(i)
                same_speaker_dst.append(j)
            # If the speaker is different, add a 'past_other' edge
            else:
                other_speaker_src.append(i)
                other_speaker_dst.append(j)
    
    # Add these new edge types to the graph for each modality
    for nt in data.node_types:
        # Add 'past_same' edges if any exist
        if same_speaker_src:
            edge_index_same = torch.tensor([same_speaker_src, same_speaker_dst], dtype=torch.long)
            data[nt, 'past_same', nt].edge_index = edge_index_same
        else:
            data[nt, 'past_same', nt].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Add 'past_other' edges if any exist
        if other_speaker_src:
            edge_index_other = torch.tensor([other_speaker_src, other_speaker_dst], dtype=torch.long)
            data[nt, 'past_other', nt].edge_index = edge_index_other
        else:
            data[nt, 'past_other', nt].edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- Cross-modal edges (unconditional) ---
    nodes = torch.arange(num_utterances)
    edge_index = torch.stack([nodes, nodes], 0)
    pairs = [("text", "audio"), ("text", "visual"), ("audio", "visual")]
    for mod_a, mod_b in pairs:
        data[mod_a, f"{mod_a}_to_{mod_b}", mod_b].edge_index = edge_index
        data[mod_b, f"{mod_b}_to_{mod_a}", mod_a].edge_index = edge_index

    data.y = labels
    data.vid = vid
    return data