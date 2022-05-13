import torch

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import LaneATT

INFINITY = 987654.

def match_proposals_with_targets(model: "LaneATT", proposals: torch.Tensor, targets: torch.Tensor, t_pos=0.04, t_neg=0.08):

    # Normalizing the x coordinates
    # TODO remove this from here, and normalize it earlier
    targets = targets.clone()
    proposals = proposals.clone()
    targets[:, 5:] = targets[:, 5:] / model.img_w
    proposals[:, 5:] = proposals[:, 5:] / model.img_w

    # repeat proposals and targets to generate all combinations
    num_proposals = proposals.shape[0]
    num_targets = targets.shape[0]
    proposals = torch.repeat_interleave(proposals, num_targets, dim=0)  # repeat_interleave'ing [a, b] 2 times gives [a, a, b, b]
    targets = torch.cat(num_proposals * [targets])  # applying this 2 times on [c, d] gives [c, d, c, d]

    # get the min start point, which means find the valid start point in image coordinate system (y pointing downwards)
    targets_starts = targets[:, 2] * model.n_strips  # n_prop * n_targets
    proposals_starts = proposals[:, 2] * model.n_strips   # n_prop * n_targets
    starts = torch.min(targets_starts, proposals_starts).round().long()

    length_delta = targets_starts - starts
    lengths = targets[:, 4] - length_delta - 1  # reduce the length by 1 because that means the number of valid offsets, but we want the end to be at the last valid index
    lengths[lengths <= 0] = 0  # a negative number here means no intersection, thus zero length
    ends = (starts - lengths).round().long()

    valid_offsets_mask = targets.new_zeros(targets.shape)  # n_prop * n_targets, 77=(5 + 72)
    all_indices = torch.arange(valid_offsets_mask.shape[0], dtype=torch.long, device=targets.device)  # 0, 1, 2, 3, 4, ... n_prop * n_targets - 1

    #   put a -1 on index `start`, giving [0, 1, 0, -1, 0]
    valid_offsets_mask[all_indices, 5 + (model.n_strips - ends)] -= 1.  # n_prop * n_targets
    #   put a 1 on index `end`, giving [0, 1, 0, 0, 0]
    valid_offsets_mask[all_indices, 5 + (model.n_strips - starts)] = 1.  # n_prop * n_targets

    valid_offsets_mask = valid_offsets_mask.cumsum(dim=1) > 0
    valid_offsets_mask[all_indices[lengths > 0], 5 + (model.n_strips - ends)[lengths > 0]] = True
    invalid_offsets_mask = ~valid_offsets_mask

    # compute distances
    # this compares [ac, ad, bc, bd], i.e., all combinations
    distances = torch.abs((targets - proposals) * valid_offsets_mask.float()).sum(dim=1) / (lengths.float() + 1e-9)
    distances[lengths <= 0] = INFINITY
    invalid_offsets_mask = invalid_offsets_mask.view(num_proposals, num_targets, invalid_offsets_mask.shape[1])
    distances = distances.view(num_proposals, num_targets)  # d[i,j] = distance from proposal i to target j

    positives = distances.min(dim=1)[0] < t_pos
    negatives = distances.min(dim=1)[0] > t_neg

    if positives.sum() == 0:
        target_positives_indices = torch.tensor([], device=positives.device, dtype=torch.long)
    else:
        target_positives_indices = distances[positives].argmin(dim=1)
    invalid_offsets_mask = invalid_offsets_mask[positives, target_positives_indices]

    return positives, invalid_offsets_mask, negatives, target_positives_indices, distances
