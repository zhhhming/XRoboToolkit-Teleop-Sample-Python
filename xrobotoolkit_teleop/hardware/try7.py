import time
from xrobotoolkit_teleop.common.xr_client import XrClient
xr_client = XrClient()
while True:
    start_time=time.time()
    xr_grip_val = xr_client.get_key_value_by_name("right_trigger")
    end_time=time.time()
    print(f"time:{end_time-start_time}")