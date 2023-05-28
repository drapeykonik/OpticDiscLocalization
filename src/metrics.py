from torch.nn.functional import pairwise_distance


def localization_accuracy(true, pred, r) -> float:
    distance = pairwise_distance(true, pred)
    accuracy = (distance <= r).sum().item() / distance.size(dim=0)
    return accuracy
