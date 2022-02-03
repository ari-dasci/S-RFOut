import flwr as fl
from typing import List, Optional, Tuple
from flwr.common import Weights
from rfout import RFOut
    
strategy = RFOut()
fl.server.start_server(config={"num_rounds": 10}, strategy = strategy)



