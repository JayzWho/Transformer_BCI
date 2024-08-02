import torch
import torch.nn.functional as F

def loglikelihood(predictions, labels):
    """
    Compute the negative log likelihood loss.
    
    Args:
    predictions (torch.Tensor): Predictions from the model of shape [batch_size, num_classes],
                                where each value is the logit or probability for each class.
    labels (torch.Tensor): Ground truth labels of shape [batch_size, num_classes], where each
                           value is the probability distribution over classes.
    
    Returns:
    torch.Tensor: The computed negative log likelihood loss.
    """
    # Ensure predictions are of type float
    predictions = predictions.float()
    
    # Ensure labels are of type float and normalized to be probabilities
    labels = labels.float()
    
    # Use log_softmax on predictions to get log probabilities
    log_probs = torch.log(predictions)
    
    # Compute the negative log likelihood loss
    loss = -torch.sum(labels * log_probs) / predictions.size(0)
    
    return loss
