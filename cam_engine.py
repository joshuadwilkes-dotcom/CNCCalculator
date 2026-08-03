import math
import cadquery as cq

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

    def estimate_layer_cam_time(self, solid, layer_z_height, removed_vol=0.0, area_scale=1.0, length_scale=1.0):
        time_outline = 0.0
        time_pocket = 0.0
        time_3d = 0.0
        unique_tools_used = set()

        # Wrap the solid in a Workplane to easily query the exact geometric faces
        wp = cq.Workplane("XY").add(solid)
        faces = wp.faces().vals()

        area_pocket = 0.0
        area_3d = 0.0

        # Feature Recognition using native B-Rep Mathematics
        for face in faces:
            geom_type = face.geomType()
            area = face.Area() * area_scale
            
            if geom_type == "PLANE" or geom_type == "CYLINDER":
                try:
                    normal = face.normalAt(face.Center())
                    z_normal = abs(normal.z)
                    
                    # Vertical wall (allows up to ~8.6 degrees of draft) or perfectly flat floor
                    if z_normal < 0.15 or z_normal > 0.99:
                        area_pocket += area
                    else:
                        # Slanted plane (true 3D surface)
                        area_3d += area
                except Exception:
                    # Fallback for highly complex faces where Center() is mathematically undefined
                    area_3d += area
            else:
                # B-Splines, Cones, Spheres, etc.
                area_3d += area
                
        # Partition the total removed volume based on exact surface area feature ratios
        total_feature_area = area_pocket + area_3d
        if total_feature_area > 0:
            vol_pocket = removed_vol * (area_pocket / total_feature_area)
            vol_3d = removed_vol * (area_3d / total_feature_area)
        else:
            vol_pocket = removed_vol
            vol_3d = 0.0
            
        # Outline Time: Calculated exactly from the layer's 2D physical bounding box
        bb = solid.BoundingBox()
        perimeter = ((bb.xlen * length_scale) * 2) + ((bb.ylen * length_scale) * 2)
        
        prof_tool = self.tools["5/8_PROFILE"]
        unique_tools_used.add("5/8_PROFILE")
        prof_passes = math.ceil(layer_z_height / prof_tool["max_stepdown"])
        ext_cut_time = (perimeter / prof_tool["cut_ipm"]) * prof_passes
        ramp_dist = layer_z_height / math.sin(math.radians(prof_tool["ramp_angle_deg"]))
        ext_ramp_time = (ramp_dist / prof_tool["ramp_ipm"]) * prof_passes
        time_outline = ext_cut_time + ext_ramp_time
        
        # Volumetric MRR for internal features
        if vol_pocket > 0:
            unique_tools_used.add("5/8_POCKET")
            time_pocket = (vol_pocket / 125.0) * 1.25 
            
        if vol_3d > 0:
            unique_tools_used.add("1/4_3D_SURFER")
            time_3d = (vol_3d / 30.0) * 1.5 
            
        # Tool swap penalties
        tool_changes = max(0, len(unique_tools_used) - 1)
        tool_swap_penalty = tool_changes * self.cfg["tool_change_penalty_min"]
        
        total_raw_time = time_outline + time_pocket + time_3d
        if total_raw_time > 0:
            time_outline += tool_swap_penalty * (time_outline / total_raw_time)
            time_pocket += tool_swap_penalty * (time_pocket / total_raw_time)
            time_3d += tool_swap_penalty * (time_3d / total_raw_time)

        return {
            "total_time_min": time_outline + time_pocket + time_3d,
            "est_outline": time_outline,
            "est_pocket": time_pocket,
            "est_3d": time_3d,
            "tool_changes": tool_changes,
            "tools_used": list(unique_tools_used)
        }
