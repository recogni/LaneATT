from lib.datasets.lane_dataset import LaneDataset
import logging
import json

def setup_logging():
    formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.DEBUG, handlers=[stream_handler])
    logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    lds = LaneDataset(split='train', root="datasets/tusimple")
    sample = lds[0]
    sample_out = {"lanes": sample[2].tolist(), "h_samples": lds.offsets_ys.tolist(), "raw_file": sample[1]}
    with open("label_laneatt.json", "w") as f:
        json.dump(sample_out, f)
