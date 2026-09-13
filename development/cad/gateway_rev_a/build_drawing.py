"""Dimensioned review drawing from the same design parameters, SVG in mm views."""
import json
from html import escape
from pathlib import Path
R=Path(__file__).resolve().parent
p=json.loads((R/'design_parameters.json').read_text());c=p['design_choices'];n=p['nvidia_reference']
a=['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1120" viewBox="0 0 1600 1120"><rect width="1600" height="1120" fill="white"/><style>text{font-family:Arial,sans-serif;fill:#172c3c}.dim{stroke:#637783;fill:none;stroke-width:1}.part{stroke:#172c3c;fill:#eef2f4;stroke-width:2}.ref{stroke:#298168;fill:none;stroke-dasharray:7 5;stroke-width:1.5}</style><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto-start-reverse"><path d="M8 0 L0 4 L8 8" fill="none" stroke="#637783"/></marker></defs>']
def text(s,x,y,size=18,anchor='start'):
    a.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}">{escape(s)}</text>')
def line(x,y,xx,yy,cl='dim'):
    a.append(f'<path d="M{x} {y} L{xx} {yy}" class="{cl}"/>')
def rect(x,y,w,h,cl='part'):
    a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="{cl}"/>')
def circle(x,y,r,color='#2468ad'):
    a.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="1.5"/>')
def dim(x,y,xx,yy,label,tx=None,ty=None):
    a.append(f'<path d="M{x} {y} L{xx} {yy}" class="dim" marker-start="url(#arrow)" marker-end="url(#arrow)"/>')
    text(label,tx if tx is not None else (x+xx)/2,ty if ty is not None else y-10,16,'middle')
text('SANASH / GATEWAY A0',55,52,30)
text('MECHANICAL REVIEW DRAWING - BENCH CANDIDATE',55,84,19)
text('All dimensions in mm | 2026-09-12 | Physical fit and thermal tests pending',55,110,16)
line(55,128,1540,128)
text('01  BASE / TOP VIEW - LID REMOVED',110,167,21)
s=3;ox=170;oy=595
def xy(x,y):return ox+x*s,oy-y*s
def planrect(x,y,w,h,cl='part'):
    xx,yy=xy(x,y+h);rect(xx,yy,w*s,h*s,cl)
def pc(x,y,r,color='#2468ad'):
    xx,yy=xy(x,y);circle(xx,yy,r*s,color)
planrect(0,0,128,118);planrect(3,3,122,112)
for x,y in c['external_mount_holes']:
    planrect(-12 if x<0 else 125,y-8,15,16);pc(x,y,2.25)
planrect(10,0,108,3)
a.append(f'<rect x="{ox+30}" y="{oy-9}" width="324" height="9" fill="white"/>')
planrect(14,24,100,79,'ref')
for i,(x,y) in enumerate(n['selected_layout_holes_xy'],1):
    x+=18;y+=41;pc(x,y,2.5,'#298168');pc(x,y,1.4);xx,yy=xy(x,y);text(f'H{i}',xx+12,yy-8,14)
for x,y in c['lid_screw_centers']:pc(x,y,5,'#172c3c');pc(x,y,1.7)
dim(ox,215,ox+384,215,'128');line(ox,215,ox,240);line(ox+384,215,ox+384,240)
dim(ox-75,241,ox-75,595,'118',ox-101,426)
dim(ox-36,654,ox+420,654,'152 including mounting ears');line(ox-36,595,ox-36,658);line(ox+420,595,ox+420,658)
text('FRONT / CABLE ACCESS',362,617,15,'middle')
text('Origin (0,0) at front-left outer corner',110,685,16)
text('Green dashed outline: carrier board 100 x 79',110,710,16)
text('H1-H4: clearance Ø2.8; stand Ø5 x 8 high',110,735,16)
text('Board underside Z=11; floor Z=3',110,760,16)
text('Board screws: M2.5 through bolt + washer + nut',110,785,16)
text('HOLE COORDINATES / X, Y',110,825,19)
for i,(x,y) in enumerate(n['selected_layout_holes_xy'],1):text(f'H{i}     {x+18:.0f}, {y+41:.0f}',110+(i-1)%2*220,858+(i-1)//2*28,17)
text('Board pitch: 86 x 58 (NVIDIA P3768 A04 layout)',110,922,16)
text('Cover: (7,7), (121,7), (7,111), (121,111)',110,949,16)
text('Mount ears: (-6,25), (134,25), (-6,93), (134,93)',110,976,16)

text('02  FRONT VIEW / COVER FITTED',865,167,21)
fx,fy=885,380
rect(fx,fy-174,384,174);rect(fx,fy-174,384,9)
rect(fx+30,fy-117,324,96)
dim(fx+425,fy-174,fx+425,fy,'58',fx+452,fy-83)
dim(fx+30,fy+30,fx+354,fy+30,'108 opening')
text('Opening: Z=7..39 (height 32)',885,449,16)
text('Walls / floor / lid: 3; base height: 55',885,475,16)
text('Installed kit envelope checked conservatively;',885,501,16)
text('factory base removed, original fan retained.',885,527,16)

text('03  LID / TOP VIEW',865,574,21)
lx,ly=885,959
rect(lx,ly-354,384,354)
for x in range(22,107,7):rect(lx+x*3,ly-300,12,204)
for x,y in c['lid_screw_centers']:circle(lx+x*3,ly-y*3,1.7*3)
text('13 slots: 4 x 68; X pitch 7; first X=22; Y=32',865,987,16)
text('4 x Ø3.4 / M3; nut pockets AF5.7 x 2.6 in base',865,1014,16)
line(55,1040,1540,1040)
text('A0: design dimensions, not measured hardware. Hole positions: NVIDIA A04; source + extraction in references/.',55,1070,16)
text('STL: base flat on floor; lid exported at Z=0. Clearances, screw lengths and print tolerances require a fit sample.',55,1097,16)
a.append('</svg>');(R/'drawings/mechanical_dimensions.svg').write_text(''.join(a),encoding='utf-8')
print('Dimension drawing created')
