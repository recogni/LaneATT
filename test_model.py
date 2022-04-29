if __name__=='__main__':
    from lib.config import Config
    cfg = Config("/home/vincentmayer/repos/LaneATT/cfgs/laneatt_tusimple_resnet122.yml")
    model = cfg.get_model()
    model.draw_anchors(640, 360)