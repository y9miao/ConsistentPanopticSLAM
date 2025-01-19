import json
import zlib
import time

import numpy as np

from rp_server import TcpServer
from rp_server import Mask2FormerWrapper
from rp_server import bin2img, bin2int, int2bin
    
def pack_mask2former_panoseg_(resp):
    """
    Pack detectron2 (Pano_seg) result data into binary

    | pkg_size (4B int) | map_size (4B int) | width (4B int) | ...
    | height (4B int) | binary_map (map_size B) | json_info_binary (rest) |
    """
    # pack segmentation map
    pano_resp = resp["panoptic_seg"]
    seg_map = pano_resp[0].cpu().detach().numpy().astype("uint8")
    h, w = seg_map.shape
    compressed_map_bin = zlib.compress(seg_map.tobytes())
    map_size_bin = int2bin( len(compressed_map_bin) )
    map_data_pkg = map_size_bin + int2bin(w) + int2bin(h) + compressed_map_bin
    
    # pack semantic information
    info_json = {
        "info": pano_resp[1],
    }
    info_bin = json.dumps(info_json).encode()

    # total package size
    print(len(map_data_pkg))
    pkg_size = int2bin(len(map_data_pkg) + len(info_bin))

    return pkg_size + map_data_pkg + info_bin
    
Mask2Former_ENCODER = {
    "Pano_seg": pack_mask2former_panoseg_,
}

class Mask2FormerServer(TcpServer):

    def __init__(self, host, port, cfg_file, model_type="Pano_seg"):
        super(Mask2FormerServer, self).__init__(host=host, port=port)
        
        self.dt_ = Mask2FormerWrapper(cfg_file, task=model_type)
        self.model_ = model_type


    def handle_connection_(self, conn, addr):
        conn_id = "{}:{}".format(addr[0], addr[1])
        print('New connection from {}'.format(conn_id))

        while not self.quit_event_.is_set():
            pack_size = conn.recv(4)
            
            # end of Connection
            if not pack_size:
                break

            pack_size = bin2int(pack_size)
            # fetch data package
            data = self.recv_all_(conn, pack_size)

            img = bin2img(data)
            ret = self.dt_.predict(img)
            
            # send back response
            conn.sendall( self.pack_(ret, self.model_) )

        conn.close()
        print("Connection {}: closed".format(conn_id))


    def pack_(self, resp, model_type):
        if model_type in Mask2Former_ENCODER:
            return Mask2Former_ENCODER[model_type](resp)
        else:
            raise Exception("Does not support type: {}".format(model_type))

    
if __name__ == "__main__":
    server = Mask2FormerServer(
        host="0.0.0.0", 
        port=8801, 
        cfg_file= "/home/yang/toolbox/test_field/Detectron2WSP/Mask2Former/configs/scannet/inference.yaml",
        model_type="Pano_seg")
    server.launch()

