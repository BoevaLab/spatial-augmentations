import copy
import torch
from torch import nn
from typing import Tuple

class BGRL(nn.Module):
    """
    BGRL (Bootstrap Graph Representation Learning) architecture for graph representation learning.

    This model is designed for self-supervised learning on graph data. It consists of an online 
    encoder, a target encoder, and a projector. The online encoder learns representations of the 
    input graph, while the target encoder provides stable targets for the online encoder to match. 
    The projector maps the online encoder's output to a space where it can be compared to the target 
    encoder's output.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        projector (torch.nn.Module): Projector network used to map the online encoder's output to a latent space.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        projector: torch.nn.Module
    ) -> None:
        """
        Initialize the BGRL model.

        This method sets up the online encoder, target encoder, and projector. The target encoder 
        is a copy of the online encoder, with its weights reinitialized and gradients disabled.

        Args:
            encoder (torch.nn.Module): The encoder network.
            projector (torch.nn.Module): The projector network.
        """
        super().__init__()

        # online and target networks
        self.online_encoder = encoder
        self.projector = projector
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights and stop gradient for the target encoder
        self.target_encoder.reset_parameters()
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        """
        Get the trainable parameters of the model.

        This method returns the parameters of the online encoder and projector, which are the 
        components of the model that are updated during training.

        Returns:
            List[torch.nn.Parameter]: A list of trainable parameters.
        """
        return list(self.online_encoder.parameters()) + list(self.projector.parameters())

    @torch.no_grad()
    def update_target_network(self, mm: float):
        """
        Update the target network using a momentum-based moving average.

        This method updates the weights of the target encoder by blending its current weights with 
        the weights of the online encoder. The blending is controlled by the momentum parameter.

        Args:
            mm (float): Momentum used in the moving average update. Must be between 0.0 and 1.0.

        Raises:
            AssertionError: If the momentum value is not in the range [0.0, 1.0].
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(
        self, 
        online_x: torch.Tensor, 
        target_x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass through the BGRL model.

        This method processes two views of the input graph through the online and target networks. 
        The online encoder's output is passed through the projector to generate predictions, while 
        the target encoder's output serves as the target for the predictions.

        Args:
            online_x (torch.Tensor): Input graph for the online encoder.
            target_x (torch.Tensor): Input graph for the target encoder.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - online_q: The predictions from the online network.
                - target_y: The target embeddings from the target network.
        """
        # forward online network
        online_y = self.online_encoder(online_x)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x).detach()
        return online_q, target_y
    

if __name__ == "__main__":
    _ = BGRL()