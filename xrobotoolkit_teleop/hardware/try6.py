import numpy as np
def normalize_angle_deg( angle_deg):
    """将任意角度归一化到 [-180, 180] 范围"""
    angle = angle_deg % 360
    if angle > 180:
        angle -= 360
    return angle
def to_nearest_equivalent_angle( target_deg, current_deg):
    """
    将目标角度调整为离当前角度最近的等效角度
    例如：current=10, target=350 -> 返回 -10 (而不是 350)
    """
    diff = target_deg - current_deg
    # 归一化差值到 [-180, 180]
    diff =normalize_angle_deg(diff)
        # 返回最近的等效角度
    return current_deg + diff
print(to_nearest_equivalent_angle(-179,180))