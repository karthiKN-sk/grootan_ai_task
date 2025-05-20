from typing import Dict, List, Iterable
import supervision as sv
import numpy as np

def initiate_polygon_zones(
    names: List[str],
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.CENTER],
) -> Dict[str, sv.PolygonZone]:
    return {
        name: sv.PolygonZone(polygon=polygon, triggering_anchors=triggering_anchors)
        for name, polygon in zip(names, polygons)
    }

def compute_centroid(polygon):
    """Compute the centroid (mean point) of a polygon."""
    centroid = np.mean(polygon, axis=0).astype(int)
    return sv.Point(*centroid)