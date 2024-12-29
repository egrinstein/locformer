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

from .srp import Srp
from datasets.array_setup import ARRAY_SETUPS


class Locformer(nn.Module):
    def __init__(
        self,
        num_samples,
        num_resolution,
        num_layers,
        dim_feedforward,
        dropout,
        num_heads=1,
        feature_extractor=None,
    ):
        super(Locformer, self).__init__()
        self.num_samples = num_samples
        self.num_resolution = num_resolution
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.feature_extractor = feature_extractor

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

    def forward(self, x, **kwargs):
        # x: (batch, num_channels, num_samples), if feature_extractor is not None
        # srp_features: (batch, num_resolution)
        if self.feature_extractor is not None:
            x = self.feature_extractor(x)

        # transformer_output: (num_resolution, batch, 3)
        transformer_output = self.transformer(x)

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


class LocformerFeatureExtractor(torch.nn.Module):
    def __init__(self, params):
        super().__init__()

        array_train = params["dataset"]["array_train"]
        array_test = params["dataset"]["array_test"]

        self.mic_pos = None
        if array_train == array_test \
            and array_train != "random":
            # If the train and test arrays are the same, the maximum delay is computed only once
            self.mic_pos = torch.from_numpy(ARRAY_SETUPS[array_train]["mic_pos"])

        win_size = params["win_size"]
        hop_rate = params["hop_rate"]
        self.c = params["speed_of_sound"]
        self.fs = params["fs"]
        self.res_phi = params["srp"]["res_phi"]

        self.srp = Srp(
            win_size,
            hop_rate,
            params["srp"]["res_the"],
            params["srp"]["res_phi"],
            self.fs,
            thetaMax=np.pi,
            mic_pos=self.mic_pos,
            gcc_transform="phat",
            normalize=True, # Verify
        )
    
    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the feature extractor
        
        Args:
            x: Tensor of shape (batch, num_channels, num_samples) 
        
        Returns:
            Tensor of shape (batch, num_resolution, 3)
        """

        x = self.srp(x)
        maps = x["signal"].unsqueeze(1)  # Add channel dimension

        maximums = maps.view(list(maps.shape[:-2]) + [-1]).argmax(dim=-1)

        max_the = (maximums / self.res_phi).float() / maps.shape[-2]
        max_phi = (maximums % self.res_phi).float() / maps.shape[-1]
        repeat_factor = np.array(maps.shape)
        repeat_factor[:-2] = 1
        repeat_factor = repeat_factor.tolist()
        maps = torch.cat(
            (
                maps,
                max_the[..., None, None].repeat(repeat_factor),
                max_phi[..., None, None].repeat(repeat_factor),
            ),
            1,
        )
        # TODO: Understand the format of the maps tensor
        # TODO: transform the maps tensor into a list of 3D vectors representing the magnitude and direction of the sound sources

        x["signal"] = maps
        return x
