import math
import numpy as np
import trimesh
from shapely.geometry import Polygon

CAM_CONFIG = {
    "tool_change_penalty_min": 1.0,
    "tools": {
        "5/8_POCKET": {
            "diameter": 0.625,
            "cut_ipm": 350.0,
            "ramp_ipm": 150.0,
            "ramp_angle_deg": 45.0,
            "max_stepdown": 1.5,
            "stepover_ratio": 0.65
        },
        "5/8_PROFILE": {
            "diameter": 0.625,
            "cut_ipm": 150.0,
            "ramp_ipm": 150.0,
            "ramp_angle_deg": 45.0,
            "max_stepdown": 1.0,
            "stepover_ratio": 1.0
        },
        "3/8_FOAM": {
            "diameter": 0.375,
            "cut_ipm": 350.0,
            "ramp_ipm": 13.333,
            "ramp_angle_deg": 89.0,
            "max_stepdown": 1.0,
            "stepover_ratio": 0.60
        },
        "1/4_3D_SURFER": {
            "diameter": 0.250,
            "cut_ipm": 150.0,
            "ramp_ipm": 150.0,
            "ramp_angle_deg": 20.0,
            "max_stepdown": 0.5,
            "lookahead_penalty": 1.35
        }
    }
}

class FoamCAMEngine:
    def __init__(self, config=CAM_CONFIG):
        self.cfg = config
        self.tools = config["tools"]

    def select_pocket_tool(self, min_slot_width):
        if min_slot_width >= 0.625:
            return "5/8_POCKET"
        elif min_slot_width >= 0.375:
            return "3/8_FOAM"
        else:
            return "1/4_3D_SURFER"

    def estimate_layer_cam_time(self, mesh, layer_z_height, is_3d_surface=False):
        try:
            slice_plane_origin = [0, 0, layer_z_height / 2.0]
            slice_2d = mesh.section(plane_origin=slice_plane_origin, plane_normal=[0, 0, 1])

            if slice_2d is None:
                return {"total_time_min": 0.0, "tool_changes": 0, "tools_used": []}

            path_2d, _ = slice_2d.to_planar()
            polygons = path_2d.polygons_full
        except Exception:
            # Fallback for complex/non-watertight slices
            return {"total_time_min": 0.0, "tool_changes": 0, "tools_used": []}

        total_cut_time = 0.0
        total_ramp_time = 0.0
        unique_tools_used = set()

        prof_tool = self.tools["5/8_PROFILE"]
        unique_tools_used.add("5/8_PROFILE")

        for poly in polygons:
            if poly is None or poly.is_empty:
                continue

            ext_length = poly.exterior.length
            prof_passes = math.ceil(layer_z_height / prof_tool["max_stepdown"])
            
            total_cut_time += (ext_length / prof_tool["cut_ipm"]) * prof_passes
            
            ramp_dist = layer_z_height / math.sin(math.radians(prof_tool["ramp_angle_deg"]))
            total_ramp_time += (ramp_dist / prof_tool["ramp_ipm"]) * prof_passes

            for interior in poly.interiors:
                pocket_poly = Polygon(interior)
                pocket_area = pocket_poly.area

                min_bounds_dim = min(pocket_poly.bounds[2] - pocket_poly.bounds[0], 
                                     pocket_poly.bounds[3] - pocket_poly.bounds[1])

                pocket_tool_key = self.select_pocket_tool(min_bounds_dim)
                p_tool = self.tools[pocket_tool_key]
                unique_tools_used.add(pocket_tool_key)

                effective_stepover = p_tool["diameter"] * p_tool["stepover_ratio"]
                clearing_distance = pocket_area / effective_stepover
                pocket_passes = math.ceil(layer_z_height / p_tool["max_stepdown"])

                pocket_cut_time = (clearing_distance / p_tool["cut_ipm"]) * pocket_passes

                pocket_ramp_dist = layer_z_height / math.sin(math.radians(p_tool["ramp_angle_deg"]))
                pocket_ramp_time = (pocket_ramp_dist / p_tool["ramp_ipm"]) * pocket_passes

                if is_3d_surface or pocket_tool_key == "1/4_3D_SURFER":
                    lookahead_multiplier = p_tool.get("lookahead_penalty", 1.35)
                    pocket_cut_time *= lookahead_multiplier

                total_cut_time += pocket_cut_time
                total_ramp_time += pocket_ramp_time

        tool_changes = max(0, len(unique_tools_used) - 1)
        tool_swap_penalty = tool_changes * self.cfg["tool_change_penalty_min"]

        raw_sim_time = total_cut_time + total_ramp_time + tool_swap_penalty

        return {
            "total_time_min": raw_sim_time,
            "tool_changes": tool_changes,
            "tools_used": list(unique_tools_used)
        }
