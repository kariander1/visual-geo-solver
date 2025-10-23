from torch.utils.data import Dataset
from data.Curves import CurveImageDataset
from data.GeoSteinerDataset import GeoSteinerDataset
from data.PolygonDataset import PolygonDataset

def get_dataset(config) -> Dataset:
    if config['dataset']['type'] == 'curves':
        dataset = CurveImageDataset(
            **config['dataset'],)
    elif config['dataset']['type'] == 'graphs_steiner':
        dataset = GeoSteinerDataset(
            **config['dataset'],)
    elif config['dataset']['type'] == 'polygons':
        dataset = PolygonDataset(
            **config['dataset'],)
    else:
        raise ValueError(f"Unsupported dataset type: {config['dataset']['type']}")
    return dataset
