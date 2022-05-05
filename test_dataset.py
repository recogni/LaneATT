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
    split = "test"
    size = (720, 1280)
    lds = LaneDataset(split=split, img_size=(720, 1280), root="datasets/tusimple" if split in ["train", "val"] else "datasets/tusimple-test")
    f = open(f"processed_{split}_{size[0]}-{size[1]}_labels_laneatt.json", "w")
    for i, sample in enumerate(lds):
        sample_out = {"lanes": sample[2].tolist(), "h_samples": lds.offsets_ys.tolist(), "raw_file": sample[1]}
        f.write(json.dumps(sample_out))
        f.write("\n")
    f.close()

