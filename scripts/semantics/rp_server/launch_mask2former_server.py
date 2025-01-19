from rp_server import Mask2FormerServer


DEFAULT_HOST = "0.0.0.0"
DT_SERVER_PORT = 8801
DT2_MODEL_TYPE = "Pano_seg"
MASK2FORMER_CFG= "./Mask2Former_scannet.yaml"

class Launcher(object):

    def __init__(
        self,
        dt_host, 
        dt_port, 
        cfg_file,
        model_type="Inst_seg"
    ):
        self.dt_server_ = Mask2FormerServer(host=dt_host, 
                                            port=dt_port, 
                                            cfg_file=cfg_file,
                                            model_type=model_type)


    def launch(self):
        self.dt_server_.launch()


if __name__ == "__main__":
    launcher = Launcher(
        dt_host=DEFAULT_HOST,
        dt_port=DT_SERVER_PORT,
        cfg_file=MASK2FORMER_CFG,
        model_type=DT2_MODEL_TYPE
    )

    launcher.launch()
