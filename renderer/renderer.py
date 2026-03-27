import os, sys, math
import ezdxf
from ezdxf import colors
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import Vec2
from shapely.geometry import box, Polygon
from shapely.ops import unary_union

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from engine.engine import (FloorPlan, Room, WallSegment,
    DoorOpening, WindowOpening, generate)

WALL_EXT_T = 0.23
WALL_INT_T = 0.115

ROOM_RGB = {
    'master_bedroom':  (232, 229, 215),  # Beige/Tan
    'bedroom_2':       (232, 229, 215),  # Beige/Tan
    'bedroom_3':       (232, 229, 215),
    'bedroom_4':       (232, 229, 215),
    'living':          (250, 250, 247),  # Almost white
    'dining':          (250, 250, 247),  # Almost white
    'kitchen':         (250, 250, 247),  # Almost white
    'toilet_attached': (212, 230, 239),  # Light Blue
    'toilet_common':   (212, 230, 239),  # Light Blue
    'utility':         (230, 222, 206),  # Tan/Light Brown
    'verandah':        (250, 250, 247),  # White
    'pooja':           (250, 235, 218),  # Peach/Cream
    'store':           (230, 222, 206),
    'staircase':       (250, 250, 247),
    'corridor':        (250, 250, 247),
}

ROOM_LABELS = {
    'master_bedroom':  'MASTER BEDROOM',
    'bedroom_2':       'BEDROOM 2',
    'bedroom_3':       'BEDROOM 3',
    'bedroom_4':       'BEDROOM 4',
    'living':          'LIVING/DINING',
    'dining':          'DINING',
    'kitchen':         'KITCHEN',
    'toilet_attached': 'ATTACHED\nTOILET',
    'toilet_common':   'COMMON\nTOILET',
    'utility':         'UTILITY',
    'verandah':        'VERANDAH',
    'pooja':           'North-East\nPUJA',
    'store':           'STORE',
    'staircase':       'STAIRCASE',
    'corridor':        'PASSAGE',
}

TOL = 0.08


def setup_doc(fp):
    """
    Creates ezdxf document with layers.
    Returns (doc, msp).
    """
    doc = ezdxf.new('R2010')
    doc.header['$INSUNITS'] = 4

    if 'DASHED' not in doc.linetypes:
        doc.linetypes.add('DASHED', pattern=[0.3, 0.2, -0.1], description='Dashed __ __ __')
    if 'CENTERLINE' not in doc.linetypes:
        # Long-short repeating pattern (common CAD-style centerline)
        doc.linetypes.add('CENTERLINE', pattern=[0.75, 0.45, -0.15, 0.15, -0.15], description='Centerline __ _ __ _')

    layers = [
        ('A-WALL',      7,   True),
        ('A-ROOM',      2,   True),
        ('FURNITURE',   9,   True),
        ('DOORS',       6,   True),
        ('WINDOWS',     5,   True),
        ('DIMENSIONS',  3,   True),
        ('ANNOTATIONS', 7,   True),
        ('BOUNDARY',    8,   True),
        ('TITLEBLOCK',  7,   True),
        ('CIRCULATION', 5,   True),
    ]
    for name, color, on in layers:
        if name in doc.layers:
            layer = doc.layers.get(name)
        else:
            layer = doc.layers.add(name)
        layer.color = color
        layer.on = on

    msp = doc.modelspace()
    return doc, msp


def _iter_polygons(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    return [poly for poly in geom.geoms if not poly.is_empty]


def _same_wall(a, b):
    return (
        abs(a.x1 - b.x1) < 1e-6 and abs(a.y1 - b.y1) < 1e-6 and
        abs(a.x2 - b.x2) < 1e-6 and abs(a.y2 - b.y2) < 1e-6 and
        abs(a.thickness - b.thickness) < 1e-6 and a.wall_type == b.wall_type
    )


def _wall_rect(wall):
    if wall.direction == 'H':
        x0, x1 = sorted((wall.x1, wall.x2))
        return box(x0, wall.y1 - wall.thickness / 2.0,
                   x1, wall.y1 + wall.thickness / 2.0)
    y0, y1 = sorted((wall.y1, wall.y2))
    return box(wall.x1 - wall.thickness / 2.0, y0,
               wall.x1 + wall.thickness / 2.0, y1)


def _opening_gap(wall, width, position, overcut=0.05):
    t = wall.thickness + overcut
    if wall.direction == 'H':
        cx = wall.x1 + position * (wall.x2 - wall.x1)
        return box(cx - width / 2.0, wall.y1 - t / 2.0,
                   cx + width / 2.0, wall.y1 + t / 2.0)
    cy = wall.y1 + position * (wall.y2 - wall.y1)
    return box(wall.x1 - t / 2.0, cy - width / 2.0,
               wall.x1 + t / 2.0, cy + width / 2.0)


def _room_side_for_wall(room, wall):
    if wall.direction == 'H':
        if abs(wall.y1 - room.y) < TOL:
            return 'S'
        if abs(wall.y1 - (room.y + room.depth)) < TOL:
            return 'N'
    else:
        if abs(wall.x1 - room.x) < TOL:
            return 'W'
        if abs(wall.x1 - (room.x + room.width)) < TOL:
            return 'E'
    return None


def _door_wall_map(fp):
    room_map = {room.room_type: room for room in fp.rooms}
    result = {room.room_type: set() for room in fp.rooms}
    for door in fp.doors:
        for name in (door.room_from, door.room_to):
            room = room_map.get(name)
            if room is None:
                continue
            side = _room_side_for_wall(room, door.wall)
            if side:
                result[name].add(side)
    return result


def _opposite(side):
    return {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}.get(side, 'S')


def _door_center(wall, position):
    if wall.direction == 'H':
        return wall.x1 + position * (wall.x2 - wall.x1), wall.y1
    return wall.x1, wall.y1 + position * (wall.y2 - wall.y1)


def _add_rect_poly(msp, x, y, w, d, layer, color=8, lineweight=13):
    pts = [(x, y), (x + w, y), (x + w, y + d), (x, y + d)]
    return msp.add_lwpolyline(
        pts,
        close=True,
        dxfattribs={'layer': layer, 'color': color, 'lineweight': lineweight},
    )


def _add_hatch_from_polygon(msp, poly, layer, color=7, rgb=None, hatch_style=1):
    hatch = msp.add_hatch(
        color=color,
        dxfattribs={'layer': layer, 'hatch_style': hatch_style},
    )
    hatch.set_solid_fill(color=color, style=hatch_style, rgb=rgb)
    hatch.paths.add_polyline_path(list(poly.exterior.coords), is_closed=True)
    for interior in poly.interiors:
        hatch.paths.add_polyline_path(list(interior.coords), is_closed=True)
    return hatch


def draw_room_fills(msp, fp):
    """
    Draws solid colour fills for each room using HATCH entities.
    Uses true colour (RGB) for room fills.
    Layer: ROOM_FILLS
    """
    for room in fp.rooms:
        rgb = ROOM_RGB.get(room.room_type, (255, 255, 255))
        hatch = msp.add_hatch(
            color=2,
            dxfattribs={'layer': 'A-ROOM',
                        'true_color': colors.rgb2int(rgb)})
        hatch.set_solid_fill(color=2, rgb=colors.RGB(*rgb))
        hatch.paths.add_polyline_path([
            (room.x, room.y),
            (room.x + room.width, room.y),
            (room.x + room.width, room.y + room.depth),
            (room.x, room.y + room.depth),
        ], is_closed=True)

def draw_walls(msp, fp):
    """
    Wall-centric drawing using Shapely union for T-junction cleanup.
    Draws all walls as ONE unified polygon per wall type.
    Uses ANSI31 hatch pattern for brick walls (TN standard).
    """
    # Step 1: build gap rectangles for door and window openings
    door_gap_polys = []
    for door in fp.doors:
        try:
            door_gap_polys.append(_opening_gap(door.wall, door.width, door.position, overcut=0.05))
        except Exception:
            pass

    win_gap_polys = []
    for win in fp.windows:
        if win.wall.wall_type != 'exterior':
            continue
        try:
            win_gap_polys.append(_opening_gap(win.wall, win.width, win.position, overcut=0.05))
        except Exception:
            pass

    all_gaps = unary_union(door_gap_polys + win_gap_polys) if (door_gap_polys or win_gap_polys) else None

    # Step 2: build wall rectangle for each WallSegment
    ext_polys = []
    int_polys = []
    for wall in fp.walls:
        r = _wall_rect(wall)
        if r is None or r.is_empty:
            continue
        if wall.wall_type == 'exterior':
            ext_polys.append(r)
        else:
            int_polys.append(r)

    # Step 3: union walls of same type into single polygon (T-junction cleanup)
    ext_mass = unary_union(ext_polys) if ext_polys else None
    int_mass = unary_union(int_polys) if int_polys else None

    # Cut openings once, after union
    if all_gaps is not None:
        if ext_mass is not None:
            ext_mass = ext_mass.difference(all_gaps)
        if int_mass is not None:
            int_mass = int_mass.difference(all_gaps)

    # Step 4: material-based hatch choice (defaults to TN brick)
    ext_pattern = 'ANSI31'
    try:
        mats = getattr(fp, 'materials', None) or []
        if mats:
            mat0 = mats[0]
            if isinstance(mat0, dict):
                name = str(mat0.get('material_name', '')).lower()
                cat = str(mat0.get('material_category', '')).lower()
                pat = str(mat0.get('hatch_pattern', '')).strip()
                if 'concrete' in name or 'concrete' in cat:
                    ext_pattern = 'solid'
                elif pat and pat.upper().startswith('ANSI'):
                    ext_pattern = pat.upper()
    except Exception:
        pass

    # Step 5: draw unified wall mass as ezdxf HATCH
    def add_wall_hatch(mass, layer, color, pattern):
        if mass is None or mass.is_empty:
            return
        polys = [mass] if mass.geom_type == 'Polygon' else list(mass.geoms)
        for poly in polys:
            if poly.is_empty:
                continue
            h = msp.add_hatch(color=color, dxfattribs={'layer': layer})
            if pattern == 'solid':
                h.set_solid_fill()
            else:
                h.set_pattern_fill(pattern, scale=0.002)
            h.paths.add_polyline_path(
                list(poly.exterior.coords),
                is_closed=True,
                flags=ezdxf.const.BOUNDARY_PATH_EXTERNAL,
            )
            for interior in poly.interiors:
                h.paths.add_polyline_path(
                    list(interior.coords),
                    is_closed=True,
                    flags=ezdxf.const.BOUNDARY_PATH_DEFAULT,
                )

    add_wall_hatch(ext_mass, 'A-WALL', 7, ext_pattern)
    add_wall_hatch(int_mass, 'A-WALL', 8, 'solid')

    # Step 6: also draw wall centrelines as LWPOLYLINE
    for wall in fp.walls:
        msp.add_lwpolyline(
            [(wall.x1, wall.y1), (wall.x2, wall.y2)],
            dxfattribs={
                'layer': 'A-WALL',
                'lineweight': 50 if wall.wall_type == 'exterior' else 25,
                'color': 7 if wall.wall_type == 'exterior' else 8,
                'linetype': 'Continuous',
            },
        )


def draw_doors(msp, fp):
    """
    Draws door symbols: door leaf line + swing arc.
    Layer: DOORS
    DATA: fp.doors
    """
    room_map = {room.room_type: room for room in fp.rooms}
    for door in fp.doors:
        wall = door.wall
        cx, cy = _door_center(wall, door.position)
        swing_room = room_map.get(door.swing_into)
        side = _room_side_for_wall(swing_room, wall) if swing_room else None

        if wall.direction == 'H':
            hinge_x = cx - door.width / 2.0 if door.hinge_side == 'left' else cx + door.width / 2.0
            hinge_y = wall.y1
            leaf_x = cx + door.width / 2.0 if door.hinge_side == 'left' else cx - door.width / 2.0
            leaf_y = wall.y1
            if side == 'S':
                arc_start, arc_end = 270, 360
                leaf_end = (hinge_x, hinge_y - door.width)
                label_pt = (cx, cy - 0.25)
            else:
                arc_start, arc_end = 0, 90
                leaf_end = (hinge_x, hinge_y + door.width)
                label_pt = (cx, cy + 0.25)
        else:
            hinge_x = wall.x1
            hinge_y = cy - door.width / 2.0 if door.hinge_side == 'left' else cy + door.width / 2.0
            leaf_x = wall.x1
            leaf_y = cy + door.width / 2.0 if door.hinge_side == 'left' else cy - door.width / 2.0
            if side == 'W':
                arc_start, arc_end = 90, 180
                leaf_end = (hinge_x - door.width, hinge_y)
                label_pt = (cx - 0.25, cy)
            else:
                arc_start, arc_end = 0, 90
                leaf_end = (hinge_x + door.width, hinge_y)
                label_pt = (cx + 0.25, cy)

        if door.door_type == 'archway':
            jamb = 0.15
            if wall.direction == 'H':
                msp.add_line((cx - door.width / 2.0, cy - jamb / 2.0), (cx - door.width / 2.0, cy + jamb / 2.0),
                             dxfattribs={'layer': 'DOORS', 'color': 7, 'lineweight': 25})
                msp.add_line((cx + door.width / 2.0, cy - jamb / 2.0), (cx + door.width / 2.0, cy + jamb / 2.0),
                             dxfattribs={'layer': 'DOORS', 'color': 7, 'lineweight': 25})
            else:
                msp.add_line((cx - jamb / 2.0, cy - door.width / 2.0), (cx + jamb / 2.0, cy - door.width / 2.0),
                             dxfattribs={'layer': 'DOORS', 'color': 7, 'lineweight': 25})
                msp.add_line((cx - jamb / 2.0, cy + door.width / 2.0), (cx + jamb / 2.0, cy + door.width / 2.0),
                             dxfattribs={'layer': 'DOORS', 'color': 7, 'lineweight': 25})
        else:
            msp.add_lwpolyline(
                [(hinge_x, hinge_y), leaf_end],
                dxfattribs={'layer': 'DOORS', 'color': 7, 'lineweight': 25},
            )
            msp.add_arc(
                center=(hinge_x, hinge_y),
                radius=door.width,
                start_angle=arc_start,
                end_angle=arc_end,
                dxfattribs={'layer': 'DOORS', 'color': 7, 'lineweight': 18},
            )

        text = door.label if door.label != 'MAIN ENTRANCE' else 'MAIN ENTRANCE'
        height = 0.18 if door.label != 'MAIN ENTRANCE' else 0.16
        msp.add_text(
            text,
            dxfattribs={'layer': 'ANNOTATIONS', 'height': height, 'color': 7},
        ).set_placement(label_pt, align=TextEntityAlignment.MIDDLE_CENTER)


def draw_windows(msp, fp):
    """
    Draws 3-line window symbol.
    Layer: WINDOWS
    DATA: fp.windows
    """
    for window in fp.windows:
        wall = window.wall
        cx, cy = _door_center(wall, window.position)
        t = wall.thickness
        if wall.direction == 'H':
            offsets = [-t / 2.0, 0.0, t / 2.0] if not window.is_ventilator else [-t / 3.0, t / 3.0]
            for offset in offsets:
                msp.add_line(
                    (cx - window.width / 2.0, wall.y1 + offset),
                    (cx + window.width / 2.0, wall.y1 + offset),
                    dxfattribs={'layer': 'WINDOWS', 'color': 5, 'lineweight': 18 if not window.is_ventilator else 13},
                )
            label_pt = (cx, cy + 0.35 if abs(wall.y1 - fp.net_d) < TOL else cy - 0.35)
        else:
            offsets = [-t / 2.0, 0.0, t / 2.0] if not window.is_ventilator else [-t / 3.0, t / 3.0]
            for offset in offsets:
                msp.add_line(
                    (wall.x1 + offset, cy - window.width / 2.0),
                    (wall.x1 + offset, cy + window.width / 2.0),
                    dxfattribs={'layer': 'WINDOWS', 'color': 5, 'lineweight': 18 if not window.is_ventilator else 13},
                )
            label_pt = (cx + 0.35 if abs(wall.x1 - fp.net_w) < TOL else cx - 0.35, cy)

        msp.add_text(
            window.label,
            dxfattribs={'layer': 'ANNOTATIONS', 'height': 0.14, 'color': 251},
        ).set_placement(label_pt, align=TextEntityAlignment.MIDDLE_CENTER)


def draw_furniture(msp, fp):
    """
    Draws detailed furniture using LWPOLYLINE rectangles and ARCs.
    Layer: FURNITURE
    DATA: fp.rooms, fp.doors (to avoid blocking doors)
    """
    rooms_iterable = fp.rooms.values() if isinstance(fp.rooms, dict) else fp.rooms
    door_wall_map = _door_wall_map(fp)

    for room in rooms_iterable:
        room_sides = door_wall_map.get(room.room_type) or {'N'}
        side = _opposite(next(iter(room_sides)))
        x0, y0, width, depth = room.x, room.y, room.width, room.depth

        if room.room_type in ('master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4'):
            # Bed outline
            bed_w = 1.9  # Standard queen/king width
            bed_d = 2.0  # Standard depth
            
            # Position bed on the opposite side of the door
            bed_x = x0 + (width - bed_w) / 2.0
            bed_y = y0 + 0.15 if side == 'S' else y0 + depth - bed_d - 0.15
            if side == 'W':
                bed_x = x0 + 0.15
                bed_y = y0 + (depth - bed_w) / 2.0
                _add_rect_poly(msp, bed_x, bed_y, bed_d, bed_w, 'FURNITURE', color=252, lineweight=13)
                _add_rect_poly(msp, bed_x, bed_y + 0.2, 0.4, 0.6, 'FURNITURE', color=251, lineweight=9) # Pillow 1
                _add_rect_poly(msp, bed_x, bed_y + bed_w - 0.8, 0.4, 0.6, 'FURNITURE', color=251, lineweight=9) # Pillow 2
                _add_rect_poly(msp, bed_x + bed_d * 0.4, bed_y + 0.1, bed_d * 0.6, bed_w - 0.2, 'FURNITURE', color=253, lineweight=9) # Blanket
            elif side == 'E':
                bed_x = x0 + width - bed_d - 0.15
                bed_y = y0 + (depth - bed_w) / 2.0
                _add_rect_poly(msp, bed_x, bed_y, bed_d, bed_w, 'FURNITURE', color=252, lineweight=13)
                _add_rect_poly(msp, bed_x + bed_d - 0.4, bed_y + 0.2, 0.4, 0.6, 'FURNITURE', color=251, lineweight=9) # Pillow 1
                _add_rect_poly(msp, bed_x + bed_d - 0.4, bed_y + bed_w - 0.8, 0.4, 0.6, 'FURNITURE', color=251, lineweight=9) # Pillow 2
                _add_rect_poly(msp, bed_x, bed_y + 0.1, bed_d * 0.6, bed_w - 0.2, 'FURNITURE', color=253, lineweight=9) # Blanket
            else:
                # N or S default
                _add_rect_poly(msp, bed_x, bed_y, bed_w, bed_d, 'FURNITURE', color=252, lineweight=13)
                # Pillows
                py = bed_y + bed_d - 0.4 if side == 'S' else bed_y
                _add_rect_poly(msp, bed_x + 0.2, py, 0.6, 0.4, 'FURNITURE', color=251, lineweight=9)
                _add_rect_poly(msp, bed_x + bed_w - 0.8, py, 0.6, 0.4, 'FURNITURE', color=251, lineweight=9)
                # Blanket
                by = bed_y if side == 'S' else bed_y + 0.4 * bed_d
                _add_rect_poly(msp, bed_x + 0.1, by, bed_w - 0.2, bed_d * 0.6, 'FURNITURE', color=253, lineweight=9)
                
            # Desk/Wardrobe
            if width > 3.0:
                _add_rect_poly(msp, x0 + 0.1, y0 + 0.1 if side == 'N' else y0 + depth - 0.7, 1.2, 0.6, 'FURNITURE', color=252, lineweight=13) # wardrobe

        elif room.room_type == 'living':
            # Sofa (L-Shape or 3+2)
            sofa_w, sofa_d = 2.2, 0.8
            sofa_x = x0 + 0.4
            sofa_y = y0 + 0.4
            
            # Main Sofa base
            _add_rect_poly(msp, sofa_x, sofa_y, sofa_w, sofa_d, 'FURNITURE', color=252, lineweight=13)
            # Cushions
            _add_rect_poly(msp, sofa_x + 0.1, sofa_y + 0.1, 0.6, 0.6, 'FURNITURE', color=251, lineweight=9)
            _add_rect_poly(msp, sofa_x + 0.8, sofa_y + 0.1, 0.6, 0.6, 'FURNITURE', color=251, lineweight=9)
            _add_rect_poly(msp, sofa_x + 1.5, sofa_y + 0.1, 0.6, 0.6, 'FURNITURE', color=251, lineweight=9)
            
            # Side Sofa
            _add_rect_poly(msp, sofa_x + sofa_w + 0.2, sofa_y, 0.8, 1.5, 'FURNITURE', color=252, lineweight=13)
            _add_rect_poly(msp, sofa_x + sofa_w + 0.3, sofa_y + 0.1, 0.6, 0.6, 'FURNITURE', color=251, lineweight=9)
            _add_rect_poly(msp, sofa_x + sofa_w + 0.3, sofa_y + 0.8, 0.6, 0.6, 'FURNITURE', color=251, lineweight=9)
            
            # Coffee Table
            _add_rect_poly(msp, sofa_x + sofa_w / 2.0 - 0.4, sofa_y + sofa_d + 0.3, 1.0, 0.6, 'FURNITURE', color=8, lineweight=13)
            
            # TV Unit
            _add_rect_poly(msp, x0 + width - 0.4, y0 + 0.4, 0.4, 1.8, 'FURNITURE', color=252, lineweight=13)

        elif room.room_type == 'dining':
            table_x = x0 + width / 2.0 - 0.6
            table_y = y0 + depth / 2.0 - 0.4
            # Dining Table
            _add_rect_poly(msp, table_x, table_y, 1.2, 0.8, 'FURNITURE', color=8, lineweight=13)
            # 6 Chairs
            cx_vals = [table_x + 0.1, table_x + 0.5, table_x + 0.9]
            for cx in cx_vals:
                _add_rect_poly(msp, cx - 0.05, table_y - 0.25, 0.4, 0.25, 'FURNITURE', color=252, lineweight=9) # Bottom row
                _add_rect_poly(msp, cx - 0.05, table_y + 0.8, 0.4, 0.25, 'FURNITURE', color=252, lineweight=9)   # Top row

        elif room.room_type == 'kitchen':
            # Dark Grey Counter L-Shape fill
            counter = 0.6
            poly1 = _add_rect_poly(msp, x0 + 0.05, y0 + depth - counter - 0.05, max(width - 0.1, 0.3), counter, 'FURNITURE', color=250, lineweight=13)
            poly2 = _add_rect_poly(msp, x0 + 0.05, y0 + 0.05, counter, max(depth - 0.1, 0.3), 'FURNITURE', color=250, lineweight=13)
            
            # Add dark granite hatch
            try:
                h1 = msp.add_hatch(color=250, dxfattribs={'layer': 'FURNITURE'})
                h1.set_solid_fill(rgb=colors.RGB(90, 95, 100))
                h1.paths.add_polyline_path(list(poly1.get_points()), is_closed=True)
                
                h2 = msp.add_hatch(color=250, dxfattribs={'layer': 'FURNITURE'})
                h2.set_solid_fill(rgb=colors.RGB(90, 95, 100))
                h2.paths.add_polyline_path(list(poly2.get_points()), is_closed=True)
            except Exception:
                pass
            
            # Sink
            sink_x = x0 + width / 2.0
            sink_y = y0 + depth - counter / 2.0 - 0.15
            _add_rect_poly(msp, sink_x, sink_y, 0.6, 0.4, 'FURNITURE', color=7, lineweight=13)
            msp.add_circle((sink_x + 0.3, sink_y + 0.2), 0.1, dxfattribs={'layer': 'FURNITURE', 'color': 7, 'lineweight': 9})
            
            # Stove
            stove_x = x0 + 0.15
            stove_y = y0 + depth / 2.0
            _add_rect_poly(msp, stove_x, stove_y, 0.4, 0.6, 'FURNITURE', color=7, lineweight=13)
            msp.add_circle((stove_x + 0.2, stove_y + 0.15), 0.12, dxfattribs={'layer': 'FURNITURE', 'color': 7, 'lineweight': 9})
            msp.add_circle((stove_x + 0.2, stove_y + 0.45), 0.12, dxfattribs={'layer': 'FURNITURE', 'color': 7, 'lineweight': 9})

        elif room.room_type in ('toilet_attached', 'toilet_common'):
            # Wash basin
            _add_rect_poly(msp, x0 + width - 0.4, y0 + 0.2, 0.4, 0.5, 'FURNITURE', color=7, lineweight=9)
            msp.add_ellipse((x0 + width - 0.2, y0 + 0.45), major_axis=(0, 0.15), ratio=0.6, dxfattribs={'layer': 'FURNITURE', 'color': 7})
            
            # WC Toilet Seat
            wc_y = y0 + depth - 0.6
            _add_rect_poly(msp, x0 + width - 0.3, wc_y, 0.3, 0.4, 'FURNITURE', color=7, lineweight=9)
            msp.add_ellipse((x0 + width - 0.4, wc_y + 0.2), major_axis=(-0.25, 0), ratio=0.6, dxfattribs={'layer': 'FURNITURE', 'color': 7})

        elif room.room_type == 'staircase':
            # Draw staircase treads
            tread_count = 8
            tread_w = (width - 0.4) / tread_count
            for i in range(tread_count):
                cx = x0 + 0.2 + i * tread_w
                msp.add_line((cx, y0 + 0.1), (cx, y0 + depth - 0.1), dxfattribs={'layer': 'FURNITURE', 'color': 8, 'lineweight': 9})
            # Draw center line and arrow
            my = y0 + depth / 2.0
            msp.add_line((x0 + 0.2, my), (x0 + width - 0.2, my), dxfattribs={'layer': 'FURNITURE', 'color': 7, 'lineweight': 13})
            msp.add_line((x0 + width - 0.2, my), (x0 + width - 0.4, my + 0.15), dxfattribs={'layer': 'FURNITURE', 'color': 7, 'lineweight': 13})
            msp.add_line((x0 + width - 0.2, my), (x0 + width - 0.4, my - 0.15), dxfattribs={'layer': 'FURNITURE', 'color': 7, 'lineweight': 13})
            msp.add_text("Up to Floor", dxfattribs={'layer': 'ANNOTATIONS', 'height': 0.15, 'color': 7}).set_placement((x0 + width / 2.0, my + 0.15), align=TextEntityAlignment.BOTTOM_CENTER)

def draw_annotations(msp, fp):
    """
    Draws room labels, dimensions, door labels.
    Layer: ANNOTATIONS
    DATA: fp.rooms
    """
    for room in fp.rooms:
        cx = room.x + room.width / 2.0
        cy = room.y + room.depth / 2.0
        label = ROOM_LABELS.get(room.room_type, room.room_type.replace('_', ' ').upper())
        height = 0.30 if room.area >= 9 else 0.22 if room.area >= 4 else 0.16
        mtext = msp.add_mtext(
            label,
            dxfattribs={
                'layer': 'ANNOTATIONS',
                'char_height': height,
                'color': 7,
                'attachment_point': 5,
            },
        )
        mtext.set_location((cx, cy + 0.12))

        dim_text = f'{room.width:.1f}m x {room.depth:.1f}m'
        dim_height = 0.18 if room.area >= 9 else 0.14
        msp.add_text(
            dim_text,
            dxfattribs={'layer': 'ANNOTATIONS', 'height': dim_height, 'color': 251},
        ).set_placement((cx, cy - 0.15), align=TextEntityAlignment.MIDDLE_CENTER)

    width_dim = msp.add_linear_dim(
        base=(fp.net_w / 2.0, fp.net_d + fp.setback_front + 1.2),
        p1=(-fp.setback_side, fp.net_d + fp.setback_front),
        p2=(fp.net_w + fp.setback_side, fp.net_d + fp.setback_front),
        text=f'{fp.plot_w:.0f}m',
        dxfattribs={'layer': 'DIMENSIONS', 'color': 251},
        override={'dimtxt': 0.18, 'dimscale': 1.5},
    )
    width_dim.render()

    depth_dim = msp.add_linear_dim(
        base=(-fp.setback_side - 1.2, fp.net_d / 2.0),
        p1=(-fp.setback_side, -fp.setback_rear),
        p2=(-fp.setback_side, fp.net_d + fp.setback_front),
        text=f'{fp.plot_d:.0f}m',
        angle=90,
        dxfattribs={'layer': 'DIMENSIONS', 'color': 251},
        override={'dimtxt': 0.18, 'dimscale': 1.5},
    )
    depth_dim.render()

    for room in fp.rooms:
        if room.area <= 4.0 or room.width < 1.2:
            continue
        dim = msp.add_linear_dim(
            base=(room.x + room.width / 2.0, room.y + room.depth + 0.3),
            p1=(room.x, room.y + room.depth),
            p2=(room.x + room.width, room.y + room.depth),
            text=f'{room.width:.1f}m',
            dxfattribs={'layer': 'DIMENSIONS', 'color': 251},
            override={'dimtxt': 0.12},
        )
        dim.render()


def draw_boundary(msp, fp):
    """
    Draws property boundary and setback lines.
    Layer: BOUNDARY
    DATA: fp.plot_w, fp.plot_d, fp.setback_front/rear/side
    """
    boundary_pts = [
        (-fp.setback_side, -fp.setback_rear),
        (fp.net_w + fp.setback_side, -fp.setback_rear),
        (fp.net_w + fp.setback_side, fp.net_d + fp.setback_front),
        (-fp.setback_side, fp.net_d + fp.setback_front),
    ]
    msp.add_lwpolyline(
        boundary_pts,
        close=True,
        dxfattribs={'layer': 'BOUNDARY', 'color': 8, 'linetype': 'DASHED', 'lineweight': 9},
    )
    msp.add_text(
        'Property Boundary',
        dxfattribs={'layer': 'BOUNDARY', 'height': 0.18, 'color': 8},
    ).set_placement((fp.net_w / 2.0, fp.net_d + fp.setback_front + 0.18), align=TextEntityAlignment.MIDDLE_CENTER)

    msp.add_line((0, fp.net_d), (fp.net_w, fp.net_d), dxfattribs={'layer': 'BOUNDARY', 'color': 8, 'linetype': 'DASHED', 'lineweight': 9})
    msp.add_line((0, 0), (fp.net_w, 0), dxfattribs={'layer': 'BOUNDARY', 'color': 8, 'linetype': 'DASHED', 'lineweight': 9})
    msp.add_line((0, 0), (0, fp.net_d), dxfattribs={'layer': 'BOUNDARY', 'color': 8, 'linetype': 'DASHED', 'lineweight': 9})
    msp.add_line((fp.net_w, 0), (fp.net_w, fp.net_d), dxfattribs={'layer': 'BOUNDARY', 'color': 8, 'linetype': 'DASHED', 'lineweight': 9})

    msp.add_text(f'Front {fp.setback_front:.1f}m', dxfattribs={'layer': 'BOUNDARY', 'height': 0.15, 'color': 8}).set_placement((fp.net_w / 2.0, fp.net_d + 0.15), align=TextEntityAlignment.MIDDLE_CENTER)
    msp.add_text(f'Rear {fp.setback_rear:.1f}m', dxfattribs={'layer': 'BOUNDARY', 'height': 0.15, 'color': 8}).set_placement((fp.net_w / 2.0, -0.20), align=TextEntityAlignment.MIDDLE_CENTER)
    msp.add_text(f'Side {fp.setback_side:.1f}m', dxfattribs={'layer': 'BOUNDARY', 'height': 0.15, 'color': 8}).set_placement((-0.25, fp.net_d / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)
    msp.add_text(f'Side {fp.setback_side:.1f}m', dxfattribs={'layer': 'BOUNDARY', 'height': 0.15, 'color': 8}).set_placement((fp.net_w + 0.25, fp.net_d / 2.0), align=TextEntityAlignment.MIDDLE_CENTER)


def draw_titleblock(msp, fp, doc):
    """
    Draws north arrow and title block.
    Layer: TITLEBLOCK
    """
    cx = fp.net_w + 3.0
    cy = fp.net_d + 3.0
    
    # North Arrow
    msp.add_circle((cx, cy), 0.6, dxfattribs={'layer': 'TITLEBLOCK', 'color': 7, 'lineweight': 13})
    msp.add_lwpolyline([(cx, cy - 0.6), (cx, cy + 0.6)], dxfattribs={'layer': 'TITLEBLOCK', 'color': 7, 'lineweight': 13})
    hatch = msp.add_hatch(color=7, dxfattribs={'layer': 'TITLEBLOCK'})
    hatch.set_solid_fill(color=7)
    hatch.paths.add_polyline_path([(cx, cy + 0.6), (cx - 0.2, cy), (cx + 0.2, cy)], is_closed=True)
    msp.add_text('N', dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.35, 'color': 7}).set_placement((cx, cy + 0.8), align=TextEntityAlignment.BOTTOM_CENTER)

    # Title Block Text
    y_title = -2.5
    full_title = f'{fp.bhk}BHK RESIDENCE FLOOR PLAN - {fp.facing.upper()} FACING PLOT: {fp.plot_w:.0f}m x {fp.plot_d:.0f}m'
    
    title_x = 0.0
    msp.add_circle((title_x - 1.0, y_title + 0.3), 0.35, dxfattribs={'layer': 'TITLEBLOCK', 'color': 7, 'lineweight': 9})
    msp.add_text(full_title, dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.45, 'color': 7}).set_placement((title_x, y_title + 0.3), align=TextEntityAlignment.BOTTOM_LEFT)
    msp.add_text(f'{fp.district}, Tamil Nadu, India', dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.3, 'color': 8}).set_placement((title_x, y_title), align=TextEntityAlignment.BOTTOM_LEFT)

    scale_y = -1.8
    for idx in range(5):
        x0 = idx * 1.0
        rgb = colors.RGB(0, 0, 0) if idx % 2 == 0 else colors.RGB(255, 255, 255)
        hatch = msp.add_hatch(color=7, dxfattribs={'layer': 'TITLEBLOCK'})
        hatch.set_solid_fill(color=7, rgb=rgb)
        hatch.paths.add_polyline_path([(x0, scale_y), (x0 + 1.0, scale_y), (x0 + 1.0, scale_y + 0.2), (x0, scale_y + 0.2)], is_closed=True)
    for idx in range(6):
        msp.add_line((idx, scale_y), (idx, scale_y + 0.25), dxfattribs={'layer': 'TITLEBLOCK', 'color': 7, 'lineweight': 13})
        label = f'{idx}' if idx < 5 else '5m'
        msp.add_text(label, dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.12, 'color': 7}).set_placement((idx, scale_y - 0.12), align=TextEntityAlignment.MIDDLE_CENTER)

    lx = fp.net_w + 1.0
    ly = fp.net_d - 1.0
    msp.add_text('LEGEND', dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.25, 'color': 7}).set_placement((lx, ly), align=TextEntityAlignment.LEFT)
    ly -= 0.35
    seen = []
    rooms_iterable = fp.rooms.values() if isinstance(fp.rooms, dict) else fp.rooms
    for room in rooms_iterable:
        if room.room_type not in seen:
            seen.append(room.room_type)
    for room_type in seen:
        rgb = ROOM_RGB.get(room_type, (255, 255, 255))
        hatch = msp.add_hatch(color=7, dxfattribs={'layer': 'TITLEBLOCK', 'true_color': colors.rgb2int(rgb)})
        hatch.set_solid_fill(color=7, rgb=colors.RGB(*rgb))
        hatch.paths.add_polyline_path([(lx, ly), (lx + 0.35, ly), (lx + 0.35, ly - 0.22), (lx, ly - 0.22)], is_closed=True)
        msp.add_text(ROOM_LABELS.get(room_type, room_type).replace('\n', ' '), dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.14, 'color': 7}).set_placement((lx + 0.55, ly - 0.11), align=TextEntityAlignment.LEFT)
        ly -= 0.32

    ly -= 0.2
    score_rows = [
        ('Vastu', fp.score_vastu),
        ('NBC', fp.score_nbc),
        ('Circulation', fp.score_circulation),
        ('Adjacency', fp.score_adjacency),
        ('Overall', fp.score_overall),
    ]
    for name, value in score_rows:
        msp.add_text(f'{name}: {value:.0%}', dxfattribs={'layer': 'TITLEBLOCK', 'height': 0.16, 'color': 7}).set_placement((lx, ly), align=TextEntityAlignment.LEFT)
        ly -= 0.22

def _mpl_rgb(rgb_tuple):
    """Convert (R,G,B) 0-255 to matplotlib 0-1 tuple."""
    return tuple(c / 255.0 for c in rgb_tuple)


def render_png_direct(fp, png_path):
    """
    Professional CAD-style floor plan renderer using matplotlib directly.
    Features: filled wall rectangles with brick hatch, room texture fills,
    furniture symbols, double-line walls, full dimension chain, D/W labels,
    north arrow, legend, and title block.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Arc, FancyArrowPatch
    import numpy as np

    BG        = '#F5F0E8'
    WALL_DARK = '#1C1C1C'
    WALL_INT  = '#383838'
    DIM_C     = '#333375'
    BOUND_C   = '#999999'
    LABEL_C   = '#111111'
    DIM_LC    = '#223388'
    LBG       = '#FDFDF5'
    DOOR_C    = '#445566'
    WIN_C     = '#3366AA'
    WIN_FILL  = '#C2D8F0'

    # Wall thicknesses (in metres, matching engine constants)
    EXT_T = WALL_EXT_T   # 0.23
    INT_T = WALL_INT_T   # 0.115

    rooms_list = list(fp.rooms.values()) if isinstance(fp.rooms, dict) else list(fp.rooms)

    LEGEND_W = 3.4
    fig_w = max(fp.plot_w + LEGEND_W + 3.0, 20)
    fig_h = max(fp.plot_d + 4.5, 14)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.axis('off')

    ox = fp.setback_side
    oy = fp.setback_rear

    # ─── Helper: dimension arrow ──────────────────────────────────────────
    def dim_arrow(x1, y1, x2, y2, label, perpdir='T', gap=0.3, fs=7.2, col=DIM_C):
        """Draw a <-> dimension annotation with label on a background box."""
        mx, my = (x1+x2)/2, (y1+y2)/2
        kw = dict(arrowstyle='<->', color=col, lw=0.9,
                  mutation_scale=8, shrinkA=0, shrinkB=0)
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=kw, zorder=30)
        # Extension ticks
        if abs(y2 - y1) < 1e-3:  # horizontal
            for xp in (x1, x2):
                ax.plot([xp, xp], [y1 - 0.06, y1 + 0.06], color=col, lw=0.8, zorder=30)
            lx, ly = mx, my + (gap if perpdir == 'T' else -gap)
        else:  # vertical
            for yp in (y1, y2):
                ax.plot([x1 - 0.06, x1 + 0.06], [yp, yp], color=col, lw=0.8, zorder=30)
            lx, ly = mx + (-gap if perpdir == 'L' else gap), my
        ax.text(lx, ly, label, ha='center', va='center', fontsize=fs,
                color=col, fontweight='bold',
                bbox=dict(fc=BG, ec='none', pad=1.2), zorder=31)

    # ─── Helper: draw wall as filled rectangle ────────────────────────────
    def draw_wall_rect(wall, zorder=5):
        t = wall.thickness
        col = WALL_DARK if wall.wall_type == 'exterior' else WALL_INT
        hatch = '//' if wall.wall_type == 'exterior' else '+'
        lw = 0.4
        if wall.direction == 'H':
            x0 = ox + min(wall.x1, wall.x2)
            y0 = oy + wall.y1 - t / 2
            w  = abs(wall.x2 - wall.x1)
            d  = t
        else:
            x0 = ox + wall.x1 - t / 2
            y0 = oy + min(wall.y1, wall.y2)
            w  = t
            d  = abs(wall.y2 - wall.y1)
        if w < 1e-4 or d < 1e-4:
            return
        # Solid fill
        p = mpatches.Rectangle((x0, y0), w, d,
                                facecolor=col, edgecolor=col, linewidth=lw, zorder=zorder)
        ax.add_patch(p)

    # ─── Property boundary ────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle((0, 0), fp.plot_w, fp.plot_d,
        lw=1.2, edgecolor=BOUND_C, facecolor='none', linestyle='--', zorder=1))
    ax.text(fp.plot_w/2, fp.plot_d + 0.2, 'Property Boundary',
            ha='center', va='bottom', fontsize=7.5, color=BOUND_C, style='italic')

    # Setback dimension lines (inner tick style)
    dim_arrow(0, fp.plot_d + 0.85, fp.plot_w, fp.plot_d + 0.85,
              f'{fp.plot_w:.0f}m', perpdir='T', gap=0.18, fs=9, col=DIM_C)
    dim_arrow(-0.9, 0, -0.9, fp.plot_d,
              f'{fp.plot_d:.0f}m', perpdir='L', gap=0.22, fs=9, col=DIM_C)
    # Front & rear setback dims
    dim_arrow(ox/2, oy, ox/2, oy+fp.net_d,
              f'', perpdir='L', gap=0.15, fs=6, col=BOUND_C)
    ax.text(fp.plot_w/2, oy/2, f'Rear / Side  {fp.setback_rear:.0f}m',
            ha='center', va='center', fontsize=6.5, color=BOUND_C, zorder=3)
    ax.text(fp.plot_w/2, fp.plot_d - fp.setback_front/2,
            f'Front  {fp.setback_front:.0f}m',
            ha='center', va='center', fontsize=6.5, color=BOUND_C, zorder=3)

    # ─── Room fills + subtle patterns ─────────────────────────────────────
    ROOM_HATCH = {
        'toilet_attached':  '...',
        'toilet_common':    '...',
        'kitchen':          '++',
        'staircase':        '////',
        'pooja':            'xx',
    }
    for room in rooms_list:
        rgb = ROOM_RGB.get(room.room_type, (248, 248, 245))
        fc  = _mpl_rgb(rgb)
        ht  = ROOM_HATCH.get(room.room_type, '')
        ec  = (0.7, 0.7, 0.7) if ht else 'none'
        p = mpatches.Rectangle(
            (ox + room.x, oy + room.y), room.width, room.depth,
            facecolor=fc, edgecolor=ec, linewidth=0.3,
            hatch=ht, zorder=2)
        ax.add_patch(p)

    # ─── Draw walls as filled rectangles ─────────────────────────────────
    for wall in fp.walls:
        draw_wall_rect(wall, zorder=5)

    # Exterior net rectangle (thick border on top of fills)
    ax.add_patch(mpatches.Rectangle(
        (ox, oy), fp.net_w, fp.net_d,
        lw=4.5, edgecolor=WALL_DARK, facecolor='none', zorder=6))

    # ─── Furniture symbols ────────────────────────────────────────────────
    # Pre-compute door swing exclusion zones per room (room-local coords)
    # so furniture is never placed inside the sweep arc of any door.
    _door_zones = {}   # room_type → list of (local_cx, local_cy, clear_r)
    for door in fp.doors:
        wall = door.wall
        dc_x = wall.x1 + door.position * (wall.x2 - wall.x1)
        dc_y = wall.y1 + door.position * (wall.y2 - wall.y1)
        # Find which room this door serves (find room that owns the wall)
        for room in rooms_list:
            rx, ry = room.x, room.y
            rw2, rd2 = room.width, room.depth
            # Door centre must be on or very near a boundary of this room
            on_h = (abs(dc_y - ry) < 0.2 or abs(dc_y - (ry + rd2)) < 0.2) and (rx - 0.05 <= dc_x <= rx + rw2 + 0.05)
            on_v = (abs(dc_x - rx) < 0.2 or abs(dc_x - (rx + rw2)) < 0.2) and (ry - 0.05 <= dc_y <= ry + rd2 + 0.05)
            if on_h or on_v:
                local_cx = dc_x - rx
                local_cy = dc_y - ry
                clear_r  = door.width + 0.3      # swing radius + 30 cm safety
                zones = _door_zones.setdefault(room.room_type, [])
                zones.append((local_cx, local_cy, clear_r))

    def _in_door_zone(room, fx, fy, fw, fd):
        """Return True if rect (fx,fy,fw,fd) in room-local coords overlaps any door zone."""
        zones = _door_zones.get(room.room_type, [])
        for (zx, zy, zr) in zones:
            # Check if any corner of the furniture rect is inside the circle
            for cx in (fx, fx + fw):
                for cy in (fy, fy + fd):
                    if (cx - zx) ** 2 + (cy - zy) ** 2 < zr ** 2:
                        return True
            # Also check if circle centre is inside the rect
            if fx <= zx <= fx + fw and fy <= zy <= fy + fd:
                return True
        return False

    def frect(x, y, w, d, fc='#DDDDCC', ec='#777766', lw=0.7, z=11, room=None):
        # x, y are absolute coords; convert to room-local if room given
        if room is not None:
            loc_x = x - (ox + room.x)
            loc_y = y - (oy + room.y)
            if _in_door_zone(room, loc_x, loc_y, w, d):
                return   # skip — would block doorway
        ax.add_patch(mpatches.Rectangle((x, y), w, d,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

    def fcircle(cx, cy, r, fc='#CCCCBB', ec='#777766', z=11):
        ax.add_patch(mpatches.Circle((cx, cy), r,
            facecolor=fc, edgecolor=ec, linewidth=0.7, zorder=z))

    for room in rooms_list:

        x0 = ox + room.x + 0.1
        y0 = oy + room.y + 0.1
        rw = room.width - 0.2
        rd = room.depth - 0.2
        rt = room.room_type

        if rt in ('master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4'):
            bw = min(rw * 0.9, 1.85)
            bd = min(rd * 0.55, 2.0)
            bx = x0 + (rw - bw) / 2
            by = y0 + rd - bd
            # Bed frame
            frect(bx, by, bw, bd, fc='#E8E2D5', ec='#665544', lw=1.0, z=11, room=room)
            # Headboard
            frect(bx, by + bd - 0.3, bw, 0.3, fc='#C8B898', ec='#665544', lw=0.8, z=12, room=room)
            # Pillows
            frect(bx + 0.12, by + bd - 0.28, bw * 0.38, 0.22, fc='#F8F5EE', ec='#AAAAAA', lw=0.5, z=13, room=room)
            frect(bx + bw * 0.5 + 0.02, by + bd - 0.28, bw * 0.38, 0.22, fc='#F8F5EE', ec='#AAAAAA', lw=0.5, z=13, room=room)
            # Side table
            frect(x0, by + bd * 0.4, 0.4, 0.4, fc='#D4C8B0', ec='#665544', lw=0.5, z=11, room=room)
            # Wardrobe
            if rw > 3.0:
                frect(x0, y0, min(rw, 1.5), 0.6, fc='#D0C4A0', ec='#665544', lw=0.8, z=11, room=room)

        elif rt in ('living', 'dining'):
            # 3-seater sofa (placed away from door side, towards opposite wall)
            sw = min(rw * 0.65, 2.2)
            sofa_y = y0 + rd - 0.85   # push sofa to far wall, away from door
            frect(x0, sofa_y, sw, 0.85, fc='#D8D0C0', ec='#776655', lw=0.9, z=11, room=room)
            frect(x0, sofa_y + 0.62, sw, 0.25, fc='#C8BCA8', ec='#776655', lw=0.5, z=12, room=room)
            # Seat cushions
            for ci in range(3):
                frect(x0 + ci * sw/3 + 0.05, sofa_y + 0.08, sw/3 - 0.1, 0.5,
                      fc='#E0D8C8', ec='#998877', lw=0.4, z=13, room=room)
            # Side sofa
            frect(x0 + sw + 0.15, sofa_y - 0.5, 0.85, min(rd * 0.45, 1.4), fc='#D8D0C0', ec='#776655', lw=0.9, z=11, room=room)
            # Coffee table (centre of room)
            frect(x0 + 0.3, sofa_y - 0.65, sw - 0.6, 0.55, fc='#B8A888', ec='#665544', lw=0.8, z=11, room=room)
            # TV unit (near entrance wall)
            frect(x0 + rw - 0.45, y0 + 0.3, 0.4, min(rd - 0.6, 1.6), fc='#C8C0B0', ec='#665544', lw=0.6, z=11, room=room)
            # Dining table (if large enough)
            if rd > 3.0 and rt == 'dining':
                tx = x0 + (rw - 1.2) / 2
                ty = y0 + 0.5
                frect(tx, ty, 1.2, 0.75, fc='#B8A878', ec='#665544', lw=0.9, z=11, room=room)
                for ci in range(3):
                    frect(tx + ci * 0.35 + 0.05, ty - 0.28, 0.28, 0.28, fc='#D4C8A8', ec='#887766', lw=0.4, z=12, room=room)
                    frect(tx + ci * 0.35 + 0.05, ty + 0.75, 0.28, 0.28, fc='#D4C8A8', ec='#887766', lw=0.4, z=12, room=room)

        elif rt == 'kitchen':
            # L-shaped counter along top and left walls (away from door)
            frect(x0, y0 + rd - 0.6, rw, 0.6, fc='#555555', ec='#333333', lw=0.8, z=11, room=room)
            frect(x0, y0, 0.6, rd - 0.6, fc='#555555', ec='#333333', lw=0.8, z=11, room=room)
            # Sink (far corner)
            frect(x0 + rw - 0.7, y0 + rd - 0.55, 0.6, 0.45, fc='#888888', ec='#555', lw=0.6, z=12)
            fcircle(x0 + rw - 0.4, y0 + rd - 0.33, 0.1, fc='#AAAAAA', z=13)
            # Stove burners
            for bx2, by2 in [(x0+0.1, y0+rd-0.52), (x0+0.1, y0+rd-0.15)]:
                fcircle(bx2 + 0.15, by2 + 0.08, 0.13, fc='#666666', ec='#444', z=12)
            # Refrigerator
            frect(x0 + rw - 0.65, y0, 0.55, 0.7, fc='#DDDDDD', ec='#999', lw=0.6, z=11, room=room)

        elif rt in ('toilet_attached', 'toilet_common'):
            # WC placed on far wall (away from door)
            wc_x = x0 + rw / 2 - 0.2; wc_y = y0 + rd - 0.55
            frect(wc_x, wc_y, 0.4, 0.22, fc='#EEEEEE', ec='#999', lw=0.6, z=12, room=room)
            ax.add_patch(mpatches.Ellipse((wc_x + 0.2, wc_y + 0.28), 0.38, 0.26,
                         facecolor='#F8F8F8', edgecolor='#AAAAAA', lw=0.7, zorder=12))
            # Wash basin (near entrance side but not blocking)
            bs_x = x0 + 0.05; bs_y = y0 + rd * 0.35
            frect(bs_x, bs_y, 0.42, 0.38, fc='#E8E8E8', ec='#888', lw=0.6, z=12, room=room)
            ax.add_patch(mpatches.Ellipse((bs_x + 0.21, bs_y + 0.19), 0.3, 0.22,
                         facecolor='#F5F5F5', edgecolor='#AAAAAA', lw=0.5, zorder=13))
            fcircle(bs_x + 0.21, bs_y + 0.19, 0.06, fc='#CCCCCC', z=14)

        elif rt == 'verandah':
            # Chairs (cane chairs)
            frect(x0 + 0.1, y0 + 0.1, 0.45, 0.45, fc='#D4C8A0', ec='#776655', lw=0.6, z=11, room=room)
            frect(x0 + rw - 0.55, y0 + 0.1, 0.45, 0.45, fc='#D4C8A0', ec='#776655', lw=0.6, z=11, room=room)
            # Small table
            frect(x0 + rw/2 - 0.22, y0 + 0.12, 0.44, 0.4, fc='#C4B890', ec='#665544', lw=0.7, z=12, room=room)

        elif rt == 'staircase':
            # Stair treads (drawn bottom-up, ascending)
            n_treads = max(8, int(rd / 0.25))
            th = rd / n_treads
            for ti in range(n_treads):
                shade = 0.88 - ti * 0.04
                frect(x0, y0 + ti * th, rw, th,
                      fc=(shade, shade, shade - 0.04), ec='#999', lw=0.3, z=11)
            # Stair arrow
            ax.annotate('', xy=(x0 + rw - 0.1, y0 + rd/2),
                        xytext=(x0 + 0.1, y0 + rd/2),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=1.2), zorder=12)
            ax.text(x0 + rw/2, y0 + rd/2 + 0.12, 'Up to Floor',
                    ha='center', fontsize=5.5, color='#333', zorder=13)

    # ─── Doors ────────────────────────────────────────────────────────────
    door_legend = {}
    for i, door in enumerate(fp.doors):
        wall = door.wall
        # Centre of door on wall
        dc_x = ox + wall.x1 + door.position * (wall.x2 - wall.x1)
        dc_y = oy + wall.y1 + door.position * (wall.y2 - wall.y1)
        hw = door.width / 2.0
        tag = f'D{i+1}'
        door_legend[tag] = door.label or 'Door'

        if wall.direction == 'H':
            # Paint wall gap white
            ax.add_patch(mpatches.Rectangle(
                (dc_x - hw, dc_y - wall.thickness/2 - 0.01), door.width, wall.thickness + 0.02,
                facecolor=BG, edgecolor='none', zorder=7))
            hx, hy = dc_x - hw, dc_y
            # Leaf line
            ax.plot([hx, hx + door.width], [hy, hy], color=DOOR_C, lw=1.5, zorder=8)
            # Arc: swing into room (approximate)
            swing_dir = 1 if door.hinge_side == 'left' else -1
            arc = Arc((hx, hy), door.width * 2, door.width * 2,
                      angle=0, theta1=0, theta2=90 * swing_dir if swing_dir > 0 else 270,
                      color=DOOR_C, lw=1.0, zorder=8)
            if swing_dir > 0:
                arc = Arc((hx, hy), door.width*2, door.width*2, angle=0, theta1=0, theta2=90, color=DOOR_C, lw=1.0, zorder=8)
            else:
                arc = Arc((hx + door.width, hy), door.width*2, door.width*2, angle=90, theta1=0, theta2=90, color=DOOR_C, lw=1.0, zorder=8)
            ax.add_patch(arc)
            ax.text(dc_x, dc_y + 0.22, tag, ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color='#222244', zorder=9,
                    bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
        else:
            ax.add_patch(mpatches.Rectangle(
                (dc_x - wall.thickness/2 - 0.01, dc_y - hw), wall.thickness + 0.02, door.width,
                facecolor=BG, edgecolor='none', zorder=7))
            hx, hy = dc_x, dc_y - hw
            ax.plot([hx, hx], [hy, hy + door.width], color=DOOR_C, lw=1.5, zorder=8)
            arc = Arc((hx, hy), door.width*2, door.width*2, angle=0, theta1=0, theta2=90, color=DOOR_C, lw=1.0, zorder=8)
            ax.add_patch(arc)
            ax.text(dc_x + 0.22, dc_y, tag, ha='left', va='center',
                    fontsize=6.5, fontweight='bold', color='#222244', zorder=9,
                    bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))

    # ─── Windows ──────────────────────────────────────────────────────────
    win_legend = {}
    for i, win in enumerate(fp.windows):
        wall = win.wall
        wc_x = ox + wall.x1 + win.position * (wall.x2 - wall.x1)
        wc_y = oy + wall.y1 + win.position * (wall.y2 - wall.y1)
        hw = win.width / 2.0
        t  = wall.thickness / 2.0
        tag = f'W{i+1}'
        win_legend[tag] = 'Window'

        if wall.direction == 'H':
            # Fill the wall opening with glazing color
            ax.add_patch(mpatches.Rectangle(
                (wc_x - hw, wc_y - t), win.width, t*2,
                facecolor=WIN_FILL, edgecolor='none', zorder=7))
            for off in (-t, 0, t):
                ax.plot([wc_x - hw, wc_x + hw], [wc_y + off, wc_y + off],
                        color=WIN_C, lw=1.4, zorder=8)
            ax.text(wc_x, wc_y + t + 0.22, tag, ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color=WIN_C, zorder=9,
                    bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))
        else:
            ax.add_patch(mpatches.Rectangle(
                (wc_x - t, wc_y - hw), t*2, win.width,
                facecolor=WIN_FILL, edgecolor='none', zorder=7))
            for off in (-t, 0, t):
                ax.plot([wc_x + off, wc_x + off], [wc_y - hw, wc_y + hw],
                        color=WIN_C, lw=1.4, zorder=8)
            ax.text(wc_x + t + 0.22, wc_y, tag, ha='left', va='center',
                    fontsize=6.5, fontweight='bold', color=WIN_C, zorder=9,
                    bbox=dict(fc='white', ec='none', pad=0.5, alpha=0.8))

    # ─── Room labels + dimensions ─────────────────────────────────────────
    for room in rooms_list:
        cx = ox + room.x + room.width / 2.0
        cy = oy + room.y + room.depth / 2.0
        label = ROOM_LABELS.get(room.room_type,
                                 room.room_type.replace('_', ' ').upper())
        area = room.width * room.depth
        fs_n = 9.0 if area >= 12 else 7.5 if area >= 6 else 6.0
        fs_d = 7.5 if area >= 12 else 6.2 if area >= 6 else 5.2

        for line_i, line in enumerate(label.split('\n')):
            yo = 0.2 * (len(label.split('\n')) - 1) / 2.0 - line_i * 0.2
            ax.text(cx, cy + 0.15 + yo, line,
                    ha='center', va='center', fontsize=fs_n,
                    fontweight='bold', color=LABEL_C, zorder=15)
        ax.text(cx, cy - 0.25, f'{room.width:.1f}m x {room.depth:.1f}m',
                ha='center', va='center', fontsize=fs_d, color=DIM_LC, zorder=15)

    # ─── Full external dimension chain ────────────────────────────────────
    # Top: overall width
    dim_arrow(0, fp.plot_d + 0.85, fp.plot_w, fp.plot_d + 0.85,
              f'{fp.plot_w:.0f}m', perpdir='T', gap=0.18, fs=9.5, col=DIM_C)
    # Left: overall height
    dim_arrow(-0.9, 0, -0.9, fp.plot_d,
              f'{fp.plot_d:.0f}m', perpdir='L', gap=0.22, fs=9.5, col=DIM_C)

    # Setback sub-dimensions (top chain below overall)
    chain_y = fp.plot_d + 0.48
    dim_arrow(0, chain_y, ox, chain_y, f'{fp.setback_side:.0f}m', perpdir='T', gap=0.12, fs=6.5, col=BOUND_C)
    dim_arrow(ox, chain_y, ox + fp.net_w, chain_y, f'{fp.net_w:.0f}m', perpdir='T', gap=0.12, fs=7.5, col=DIM_C)
    dim_arrow(ox + fp.net_w, chain_y, fp.plot_w, chain_y, f'{fp.setback_side:.0f}m', perpdir='T', gap=0.12, fs=6.5, col=BOUND_C)

    # Left chain: setback heights
    chain_x = -0.52
    dim_arrow(chain_x, 0, chain_x, oy, f'{fp.setback_rear:.0f}m', perpdir='L', gap=0.14, fs=6.5, col=BOUND_C)
    dim_arrow(chain_x, oy, chain_x, oy + fp.net_d, f'{fp.net_d:.0f}m', perpdir='L', gap=0.14, fs=7.5, col=DIM_C)
    dim_arrow(chain_x, oy + fp.net_d, chain_x, fp.plot_d, f'{fp.setback_front:.0f}m', perpdir='L', gap=0.14, fs=6.5, col=BOUND_C)

    # Per-room horizontal dims (inside room top edge)
    for room in rooms_list:
        if room.width * room.depth < 4.0:
            continue
        rx, ry = ox + room.x, oy + room.y
        dim_arrow(rx + 0.05, ry + room.depth + 0.28,
                  rx + room.width - 0.05, ry + room.depth + 0.28,
                  f'{room.width:.1f}m', perpdir='T', gap=0.12, fs=6.0, col='#446688')

    # ─── North Arrow ──────────────────────────────────────────────────────
    na_x = fp.plot_w + LEGEND_W * 0.45
    na_y = fp.plot_d + 0.3
    R = 0.62
    ax.add_patch(mpatches.Circle((na_x, na_y), R, fill=False,
                                  edgecolor='#111', lw=1.8, zorder=25))
    ax.plot([na_x, na_x], [na_y - R * 0.7, na_y + R * 0.85],
            color='#111', lw=1.5, zorder=26)
    ax.fill([na_x - 0.18, na_x, na_x + 0.18],
            [na_y, na_y + R * 0.92, na_y], color='#111', zorder=27)
    ax.text(na_x, na_y + R + 0.15, 'N', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#111', zorder=28)

    # ─── Legend panel ─────────────────────────────────────────────────────
    lx0 = fp.plot_w + ox + 0.45
    ly0 = fp.plot_d - 0.35
    leg_w = LEGEND_W - 0.7

    # Gather unique room types
    seen, leg_items = [], []
    for room in rooms_list:
        if room.room_type not in seen:
            seen.append(room.room_type)
            rgb = ROOM_RGB.get(room.room_type, (248, 248, 245))
            abbr = {'master_bedroom': 'BR1', 'bedroom_2': 'BR2', 'bedroom_3': 'BR3',
                    'bedroom_4': 'BR4', 'living': 'Dwg', 'dining': 'Din',
                    'kitchen': 'Kit', 'toilet_attached': 'Tlt1',
                    'toilet_common': 'Tlt2', 'utility': 'Utl',
                    'verandah': 'Vda', 'pooja': 'Pja', 'staircase': 'Str'
                    }.get(room.room_type, room.room_type[:4].title())
            full_name = ROOM_LABELS.get(room.room_type,
                        room.room_type.replace('_', ' ').title())
            leg_items.append((abbr, full_name.replace('\n', '/'), rgb))

    leg_h = len(leg_items) * 0.37 + 0.8
    ax.add_patch(mpatches.FancyBboxPatch(
        (lx0 - 0.12, ly0 - leg_h), leg_w, leg_h,
        boxstyle='round,pad=0.1', facecolor=LBG, edgecolor='#BBBBBB', lw=0.9, zorder=18))
    ax.text(lx0 + leg_w/2 - 0.12, ly0 - 0.08, 'LEGEND',
            ha='center', va='top', fontsize=9, fontweight='bold', color='#111', zorder=19)
    ax.plot([lx0 - 0.1, lx0 + leg_w - 0.15],
            [ly0 - 0.42, ly0 - 0.42], color='#AAAAAA', lw=0.7, zorder=19)

    col_abbr_x = lx0 + 0.02
    col_swatch_x = lx0 + 0.55
    col_name_x = lx0 + 0.95

    for ki, (abbr, full_name, rgb) in enumerate(leg_items):
        iy = ly0 - 0.55 - ki * 0.37
        ax.text(col_abbr_x, iy, abbr, va='center', fontsize=7,
                fontweight='bold', color='#333366', zorder=19)
        ax.add_patch(mpatches.Rectangle((col_swatch_x, iy - 0.12), 0.33, 0.27,
            facecolor=_mpl_rgb(rgb), edgecolor='#777', lw=0.6, zorder=19))
        ax.text(col_name_x, iy, full_name[:16], va='center',
                fontsize=6.5, color='#222', zorder=19)

    # Symbols sub-section
    sym_y = ly0 - leg_h - 0.5
    ax.add_patch(mpatches.FancyBboxPatch(
        (lx0 - 0.12, sym_y - (len(door_legend) + len(win_legend)) * 0.3 - 0.55),
        leg_w, (len(door_legend) + len(win_legend)) * 0.3 + 0.55,
        boxstyle='round,pad=0.1', facecolor=LBG, edgecolor='#BBBBBB', lw=0.9, zorder=18))
    ax.text(lx0 + leg_w/2 - 0.12, sym_y - 0.08, 'Symbols',
            ha='center', va='top', fontsize=9, fontweight='bold', color='#111', zorder=19)
    ax.plot([lx0 - 0.1, lx0 + leg_w - 0.15],
            [sym_y - 0.42, sym_y - 0.42], color='#AAAAAA', lw=0.7, zorder=19)

    all_syms = list(door_legend.items()) + list(win_legend.items())
    for si, (tag, lbl) in enumerate(all_syms[:12]):
        sy = sym_y - 0.55 - si * 0.3
        color = '#333366' if tag.startswith('D') else WIN_C
        ax.text(lx0 + 0.02, sy, tag, fontsize=7, fontweight='bold', color=color, va='center', zorder=19)
        ax.text(lx0 + 0.62, sy, lbl[:20], fontsize=6.2, color='#333', va='center', zorder=19)

    # Wall info box
    info_y = sym_y - (len(door_legend) + len(win_legend)) * 0.3 - 1.1
    ax.text(lx0 + leg_w/2 - 0.12, info_y,
            f'Ext. Wall : 230 mm\nInt. Wall  : 115 mm\n\nNet Area:\n{fp.net_w:.1f}m × {fp.net_d:.1f}m\n\nScale  1:50',
            ha='center', va='top', fontsize=7.5, color='#333', linespacing=1.5,
            bbox=dict(fc=LBG, ec='#BBBBBB', pad=6, boxstyle='round'), zorder=19)

    # ─── Title block ──────────────────────────────────────────────────────
    facing_full = {'N':'NORTH','S':'SOUTH','E':'EAST','W':'WEST'}.get(
        fp.facing.upper(), fp.facing.upper())
    title = (f'{fp.bhk}BHK RESIDENCE FLOOR PLAN  –  '
             f'{facing_full} FACING PLOT: {fp.plot_w:.0f}m × {fp.plot_d:.0f}m')

    ax.plot([0, fp.plot_w + LEGEND_W], [-0.55, -0.55],
            color='#333333', lw=2.2, zorder=18)
    ax.plot([0, fp.plot_w + LEGEND_W], [-0.58, -0.58],
            color='#333333', lw=0.8, zorder=18)
    ax.text(fp.plot_w / 2, -0.72, title, ha='center', va='top',
            fontsize=13, fontweight='bold', color='#111', zorder=20)
    ax.text(fp.plot_w / 2, -1.35, f'{fp.district}, Tamil Nadu, India',
            ha='center', va='top', fontsize=9, color='#444', zorder=20)

    ax.set_xlim(-1.4, fp.plot_w + LEGEND_W + 0.6)
    ax.set_ylim(-2.2, fp.plot_d + 1.8)

    fig.savefig(png_path, dpi=150, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    print(f"  PNG exported (architectural): {png_path}")


def render(fp: FloorPlan,
           output_dir: str = 'outputs') -> dict:
    """
    Main render function.
    Returns dict: {'dxf': path, 'png': path}
    """
    orig_rooms = fp.rooms
    rooms_list = list(fp.rooms.values()) if isinstance(fp.rooms, dict) else list(fp.rooms)
    fp.rooms = rooms_list

    try:
        os.makedirs(output_dir, exist_ok=True)

        # ── Save DXF (CAD format) ──────────────────────────────────────────
        doc, msp = setup_doc(fp)
        draw_boundary(msp, fp)
        draw_room_fills(msp, fp)
        draw_furniture(msp, fp)
        draw_walls(msp, fp)
        draw_doors(msp, fp)
        draw_windows(msp, fp)
        draw_annotations(msp, fp)
        draw_titleblock(msp, fp, doc)

        dxf_path = os.path.join(
            output_dir,
            f'plan_{fp.district}_{fp.bhk}BHK_{fp.facing}.dxf')
        doc.saveas(dxf_path)

        # ── Save PNG via direct matplotlib renderer ────────────────────────
        png_path = dxf_path.replace('.dxf', '.png')
        render_png_direct(fp, png_path)

        return {'dxf': dxf_path, 'png': png_path}
    finally:
        fp.rooms = orig_rooms


if __name__ == '__main__':
    cases = [
        {'plot_w': 12, 'plot_d': 15, 'bhk': 2,
         'facing': 'N', 'district': 'Coimbatore'},
        {'plot_w': 15, 'plot_d': 20, 'bhk': 3,
         'facing': 'N', 'district': 'Chennai'},
        {'plot_w': 20, 'plot_d': 25, 'bhk': 4,
         'facing': 'N', 'district': 'Madurai'},
    ]
    os.makedirs('outputs', exist_ok=True)
    for p in cases:
        print(f"Rendering {p['plot_w']}x{p['plot_d']} {p['bhk']}BHK {p['district']}...")
        fp = generate(p)
        results = render(fp, output_dir='outputs')
        print(f"  DXF: {results['dxf']}")
        print(f"  PNG: {results['png']}")
        print(f"  Rooms:{len(fp.rooms)} Walls:{len(fp.walls)} Doors:{len(fp.doors)} Windows:{len(fp.windows)}")


