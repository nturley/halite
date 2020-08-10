import xml.etree.ElementTree as ET
from IPython.display import SVG, display
from dataclasses import dataclass
from ipywidgets import interact
import ipywidgets as widgets

@dataclass
class Dimension:
    width: int
    height: int
    
    def __mul__(self, other):
        return Dimension(self.width * other.width, self.height * other.height)
    
    def __add__(self, other):
        return Dimension(self.width + other.width, self.height + other.height)

@dataclass(frozen=True)
class DPoint:
    x: int
    y: int
        
    def __repr__(self):
        return f'({self.x}, {self.y})'
    
    def scale(self, mult: Dimension, plus=Dimension(0, 0)):
        return DPoint(
            self.x * mult.width + plus.width,
            self.y * mult.height + plus.height
        )


colors = ['red', 'green', 'blue', 'orange']
letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*_-+=`~,.?/:;"{}[]|0123456789'
def draw_board(board, highlights=[], sequence=[], cell_halite_enable=True):
    board_size = board.configuration.size
    board_dim = Dimension(board_size, board_size)
    margin_dim = Dimension(20, 20)
    node_dim = Dimension(40, 40)
    half_node_dim = node_dim * Dimension(0.5, 0.5)
    stroke_size = Dimension(1, 1)
    side_bar = Dimension(45, 0)
    side_bar_row_height = Dimension(1, 20)
    tiles_dim = board_dim * node_dim + margin_dim + stroke_size
    svg_dims =  tiles_dim + side_bar+Dimension(100, 0)

    svg = ET.Element(
        'svg',
        xmlns="http://www.w3.org/2000/svg",
        version="1.1",
        height=str(svg_dims.height),
        width=str(svg_dims.width))
    svg.append(ET.fromstring('''
        <style>
            line { stroke: black }
            
        </style>'''))
    
    for x in range(board_dim.width):
        p = DPoint(x, 0).scale(node_dim, margin_dim + half_node_dim)
        svg.append(ET.fromstring(f'<text x="{p.x}" y="{margin_dim.height - 7}" text-anchor="middle" font-weight="bold">{x}</text>'))
    
    for y in range(board_dim.height):
        p = DPoint(0, y).scale(node_dim, margin_dim + half_node_dim)
        svg.append(ET.fromstring(f'<text x="{margin_dim.width - 7}" y="{p.y}" text-anchor="end" alignment-baseline="middle" font-weight="bold">{y}</text>'))
    
    for highlight in highlights:
        p = DPoint(highlight.x, highlight.y).scale(node_dim, margin_dim)
        svg.append(ET.fromstring(f'<rect x="{p.x}" y="{p.y}" width="{node_dim.width}" height="{node_dim.height}" fill="yellow" />'))

    # draw dashed gray grid
    for x in range(board_dim.width + 1):
        p = DPoint(x, 0).scale(node_dim, margin_dim)
        dashed = 'stroke-dasharray="4 8"' if x > 0 else ''
        svg.append(ET.fromstring(f'<line x1="{p.x}" x2="{p.x}" y1="5" y2="{tiles_dim.height}" {dashed}/>'))
    for y in range(board_dim.height + 1):
        p = DPoint(0, y).scale(node_dim, margin_dim)
        dashed = 'stroke-dasharray="4 8"' if y > 0 else ''
        svg.append(ET.fromstring(f'<line x1="5" x2="{tiles_dim.width}" y1="{p.y}" y2="{p.y}" {dashed}/>'))
    
    
        
    # draw cell halite
    if cell_halite_enable:
        for cell in board.cells.values():
            p = DPoint(cell.position.x, cell.position.y).scale(node_dim, margin_dim + node_dim * Dimension(0.5, 0.3))
            svg.append(ET.fromstring(f'<text x="{p.x}" y="{p.y}" text-anchor="middle">{cell.halite:.0f}</text>'))
        
    # shipyards
    for shipyard in board.shipyards.values():
        p = DPoint(shipyard.position.x, shipyard.position.y).scale(node_dim, margin_dim+Dimension(11,15))
        svg.append(ET.fromstring(f'<rect x="{p.x}" y="{p.y}" width="18" height="18" fill="{colors[shipyard.player_id]}" />'))
    
    # draw ship positions
    for i, ship in enumerate(board.ships.values()):
        sid = letters[int(ship.id.split('-')[0])]
        p = DPoint(ship.position.x, ship.position.y).scale(node_dim, margin_dim + node_dim * Dimension(0.5, 0.6))
        svg.append(ET.fromstring(f'<circle cx="{p.x}" cy="{p.y}" r="9" fill="{colors[ship.player_id]}" />'))
        svg.append(ET.fromstring(f'<text x="{p.x}" y="{p.y+5}" text-anchor="middle" fill="white">{sid}</text>'))
        p = DPoint(20, i).scale(side_bar_row_height, Dimension(tiles_dim.width, 10))
        svg.append(ET.fromstring(f'<circle cx="{p.x}" cy="{p.y}" r="9" fill="{colors[ship.player_id]}" />'))
        svg.append(ET.fromstring(f'<text x="{p.x}" y="{p.y+5}" text-anchor="middle" fill="white">{sid}</text>'))
        svg.append(ET.fromstring(f'<text x="{p.x+12}" y="{p.y+5}" >{ship.halite}</text>'))
    
    for player in board.players.values():
        p = DPoint(20, player.id).scale(side_bar_row_height, Dimension(tiles_dim.width+side_bar.width, 15))
        svg.append(ET.fromstring(f'<text x="{p.x}" y="{p.y}" fill="{colors[player.id]}">{colors[player.id][0].upper()}: {player.halite}</text>'))
    
    for p1, p2 in zip(sequence[:-1], sequence[1:]):
        dp1 = DPoint(p1.x, p1.y).scale(node_dim, margin_dim + node_dim * Dimension(0.5, 0.6))
        dp2 = DPoint(p2.x, p2.y).scale(node_dim, margin_dim + node_dim * Dimension(0.5, 0.6))
        svg.append(ET.fromstring(f'<line x1="{dp1.x}" y1="{dp1.y}" x2="{dp2.x}" y2="{dp2.y}" />'))
    
    return SVG(ET.tostring(svg))

def draw_game(boards, boards_highlights=None):
    if boards_highlights is None:
        boards_highlights = [[]]*len(boards)
    def display_boards(t, cell_halite_enable):
        return draw_board(boards[t], boards_highlights[t], [], cell_halite_enable)
    interact(display_boards, t=widgets.IntSlider(min=0, max=len(boards)-1, step=1, value=0), cell_halite_enable=True)
