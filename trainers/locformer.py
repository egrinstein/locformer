from datasets.tau_nigens_dataset import TauNigensDataLoader
import torch

import torch.optim as optim

from tqdm import tqdm

from models.locformer import Locformer, LocformerFeatureExtractor
from utils import dict_to_device, dict_to_float, get_device


class LocformerTrainer(torch.nn.Module):
    def __init__(self, params, loss, print_model=True, allow_mps=True):
        super().__init__()
        self.device = get_device(allow_mps)

        self.params = params
        self.loss = loss
        self.sanity_check_mode: bool = params["sanity_check"]

        # create model
        model = Locformer(params["nb_gcc_bins"], params["locformer"]).to(self.device)
        self.model = model

        # Only used during inference
        self.feature_extractor = LocformerFeatureExtractor(params).to(self.device)

        self.optimizer = optim.Adam(model.parameters(), lr=params["training"]["lr"])

        if params["model_checkpoint_path"] != "":
            self.load_checkpoint(params["model_checkpoint_path"])

        if print_model:
            print(model)

    def load_checkpoint(self, path):
        print(f"Loading model from checkpoint {path}")
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)

    def _epoch(
        self, data_generator: TauNigensDataLoader, test: bool = False, metric=None
    ):
        nb_batches = 0
        loss_epoch = 0.0

        if test:
            self.model.eval()
        else:
            self.model.train()

        for data, target in tqdm(data_generator.get_batch(), total=len(data_generator)):
            # load one batch of data

            data = dict_to_float(dict_to_device(data, self.device))
            target = target.to(self.device).float()
            # process the batch of data based on chosen mode

            if test:
                with torch.no_grad():
                    output = self.model(data)
            else:
                self.optimizer.zero_grad()
                output = self.model(data)

            loss_batch = self.loss(output, target)

            if not test:
                loss_batch.backward()
                self.optimizer.step()

            if metric is not None:
                metric.partial_compute_metric(target, output)

            loss_epoch += loss_batch.item()

            nb_batches += 1
            if self.sanity_check_mode and nb_batches == 4:
                break

        loss_epoch /= nb_batches

        return loss_epoch

    def train_epoch(self, data_generator):
        return self._epoch(data_generator, "train")

    def test_epoch(self, data_generator, metric=None):
        return self._epoch(data_generator, "test", metric)

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.model(x)
