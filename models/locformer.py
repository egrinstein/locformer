# Locformer model, a Sound Source Localization (SSL) which uses the attention mechanism
# to localize the sound source in the 3D space. The model expects a matrix of dimensions (num_channels, num_samples),
# from which it extracts the Steered Response Power (SRP) features for num_resolution candidate locations. The final input
# features are obtained by weighting the num_resolution unit vectors of the candidate locations with their
# corresponding SRP features. The features are then passed through a transformer encoder to obtain the final output.
# The model outputs a vector, also of dimensions (num_resolution), where each element represents the probability of the
# sound source being at the corresponding candidate location. Note that multiple sound sources can be localized by the model.

from typing import Literal
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

from .srp import SRP


class LocFormer(nn.Module):
    def __init__(
        self,
        num_samples,
        num_resolution,
        num_layers,
        dim_feedforward,
        dropout,
        num_heads=1,
    ):
        super(LocFormer, self).__init__()
        self.num_samples = num_samples
        self.num_resolution = num_resolution
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # SRP layer
        self.srp = SRP(num_resolution)

        # Transformer model, which computes the attention mechanism on the SRP-weighted candidate locations of shape (num_resolution, 3)
        # to generate the final output of same shape.
        self.transformer = nn.TransformerEncoder
        (
            nn.TransformerEncoderLayer(
                d_model=3,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            ),
            num_layers,
        )

    def forward(self, x):
        # x: (batch, num_channels, num_samples)
        # srp_features: (batch, num_resolution)
        srp_features = self.srp(x)

        # candidate_locations: (num_resolution, 3)
        candidate_locations = self.srp.candidate_locations

        # srp_features: (batch, num_resolution, 1)
        srp_features = rearrange(srp_features, "b r -> b r ()")

        # directional SRP: (batch, num_resolution, 3)
        directional_srp = srp_features * candidate_locations

        # transformer_input: (num_resolution, batch, 3)
        transformer_input = rearrange(directional_srp, "b r d -> r b d")

        # transformer_output: (num_resolution, batch, 3)
        transformer_output = self.transformer(transformer_input)

        # output: (batch, num_resolution)
        output = rearrange(transformer_output, "r b d -> b r")

        return output


def estimate_locations_from_output(
    network_output: torch.Tensor,
    prob_threshold: float = 0.5,
    angle_threshold: float = 8.0,
):
    """
    Estimate the number and location of the sound sources from the network output, similarly to the thresholding procedure
    described in the ICRA paper "Deep Neural Networks for Multiple Speaker Detection and Localization", by He et al. (2018). A sound source
    is detected if the probability of the sound source being at a location is greater than the prob_threshold, its probability is bigger than
    other locations within the angle_threshold.

    Args:
        network_output: Tensor of shape (batch, num_resolution, 3). The direction of the vectors represents the
        direction of the sound source, and the magnitude of the vector represents the probability of the sound source
        being at that location.
        prob_threshold: The minimum probability threshold for a sound source to be detected.
        angle_threshold: The minimum angular distance in degrees between sources for them to be considered distinct.

    """
    batch_size, num_resolution, _ = network_output.size()

    # Get the probabilities
    probabilities = torch.norm(network_output, p=2, dim=-1)

    # Normalize the vectors
    locations = network_output / probabilities.unsqueeze(-1)

    # Initialize the output
    output = []

    # Iterate over the batch
    for i in range(batch_size):
        # Get the probabilities and locations for the current batch
        probs_i = probabilities[i]
        locations_i = locations[i]
        output_i = []

        # Iterate over the location
        for j in range(num_resolution):
            # If the probability is above the threshold
            if probs_i[j] < prob_threshold:
                continue

            # Get neighbors as a boolean mask
            neighbors = angular_distance(locations_i[j].unsqueeze(0), locations_i, "degrees") < angle_threshold
            # Apply the mask to the probabilities
            print(probs_i[neighbors], probs_i[neighbors].shape)
            max_neighbors_probs_i = probs_i[neighbors].max()
            
            # If the probability is the highest in the neighborhood
            if probs_i[j] >= max_neighbors_probs_i:
                output_i.append(locations_i[j])
            
        output.append(output_i)

    return output

            
def angular_distance(a: torch.Tensor, b: torch.Tensor, output_mode=Literal["radians", "degrees"]):
    """
    Compute the angular distance between two vectors
    """

    if len(a.shape) != len(b.shape):
        raise ValueError("The input tensors must have the same rank")
    
    dot = torch.sum(a * b, dim = -1)
    output = torch.acos(dot)

    if output_mode == "degrees":
        output = output * 180 / np.pi
    
    return output
