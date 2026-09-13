"""Run with FreeCAD's bundled python.exe. Native editable CSG + spreadsheet.

The factory plastic base must be removed; stock heatsink/fan retained.
Blue reference shapes are clearance envelopes, NOT detailed NVIDIA CAD.
"""
import json
import math
from pathlib import Path
import FreeCAD as App
import Part
import MeshPart

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'design_parameters.json').read_text())
C=P['design_choices']; N=P['nvidia_reference']
doc=App.newDocument('SANASH_Gateway_A0')
params=doc.addObject('Spreadsheet::Sheet','Parameters')
rows=[('Width',C['outer_width']),('Depth',C['outer_depth']),('Height',C['base_height']),('Wall',C['wall']),('Lid',C['lid_thickness']),('BoardX',C['board_origin_x']),('BoardY',C['board_origin_y']),('Stand',C['standoff_height']),('BoardT',N['board_thickness']),('TopKeepout',C['upper_keepout_height'])]
for row,(name,value) in enumerate(rows,1):
    params.set(f'A{row}',name);params.set(f'B{row}',str(value)+' mm');params.setAlias(f'B{row}',name)
params.set('A12','Dimensions in mm; blue shapes are clearance envelopes')
params.set('A13','Factory base removed. Stock heatsink/fan retained.')
params.setColumnWidth('A',350)
construction=doc.addObject('App::DocumentObjectGroup','Construction')
refs=doc.addObject('App::DocumentObjectGroup','ReferenceGeometry')
def box(name,x,y,z,dx,dy,dz):
    o=doc.addObject('Part::Box',name)
    for prop,v in [('Length',dx),('Width',dy),('Height',dz),('Placement.Base.x',x),('Placement.Base.y',y),('Placement.Base.z',z)]:
        if isinstance(v,str):o.setExpression(prop,v)
        elif prop.startswith('Placement'):setattr(o.Placement.Base,prop[-1],v)
        else:setattr(o,prop,v)
    # FreeCAD returns a copy of Placement.Base, so assign translation explicitly.
    o.Placement.Base=App.Vector(0 if isinstance(x,str) else x,0 if isinstance(y,str) else y,0 if isinstance(z,str) else z)
    for prop,v in [('Placement.Base.x',x),('Placement.Base.y',y),('Placement.Base.z',z)]:
        if isinstance(v,str):o.setExpression(prop,v)
    construction.addObject(o);return o
def cyl(name,x,y,z,r,h):
    o=doc.addObject('Part::Cylinder',name);o.Radius=r;o.Height=h;o.Placement.Base=App.Vector(x,y,z);construction.addObject(o);return o
def multi(name,shapes):
    o=doc.addObject('Part::MultiFuse',name);o.Shapes=shapes;construction.addObject(o);return o
def cut(name,base,tool):
    o=doc.addObject('Part::Cut',name);o.Base=base;o.Tool=tool;o.Refine=True;construction.addObject(o);return o

outer=box('OuterShell',0,0,0,'Parameters.Width','Parameters.Depth','Parameters.Height')
inner=box('InnerVoid','Parameters.Wall','Parameters.Wall','Parameters.Wall','Parameters.Width - 2 * Parameters.Wall','Parameters.Depth - 2 * Parameters.Wall','Parameters.Height')
shell=cut('HollowShell',outer,inner)
front=box('CableAccess',*C['front_opening'])
vents=[]
# Horizontal side and rear slots leave 3 mm ribs. These are editable primitives.
for z in [18,25,32,39]:
    for side,x in [('Left',-1),('Right',124)]:
        vents.append(box(f'{side}Vent{z}',x,41,z,5,53,4))
    vents.append(box(f'RearVent{z}',25,114,z,78,5,4))
opened=cut('VentilatedShell',shell,multi('ShellOpenings',[front]+vents))

board_holes=[(C['board_origin_x']+x+4,C['board_origin_y']+y+17) for x,y in N['selected_layout_holes_xy']]
stands=[cyl(f'BoardStand{i}',x,y,3,C['standoff_od']/2,C['standoff_height']) for i,(x,y) in enumerate(board_holes,1)]
posts=[cyl(f'LidPost{i}',x,y,3,5,52) for i,(x,y) in enumerate(C['lid_screw_centers'],1)]
tabs=[box(f'MountEar{i}',x-6,y-8,0,12+3,16,5) if x<0 else box(f'MountEar{i}',125,y-8,0,15,16,5) for i,(x,y) in enumerate(C['external_mount_holes'],1)]
reinforced=multi('ShellWithMounts',[opened]+stands+posts+tabs)
drills=[cyl(f'BoardDrill{i}',x,y,-1,C['board_screw_clearance']/2,13) for i,(x,y) in enumerate(board_holes,1)]
drills += [cyl(f'CoverDrill{i}',x,y,40,C['lid_screw_clearance']/2,20) for i,(x,y) in enumerate(C['lid_screw_centers'],1)]
drills += [cyl(f'EarDrill{i}',x,y,-1,C['external_mount_clearance']/2,7) for i,(x,y) in enumerate(C['external_mount_holes'],1)]
# Real Part::Prism primitives keep hex pockets directly editable in FreeCAD.
for i,(x,y) in enumerate(C['lid_screw_centers'],1):
    h=doc.addObject('Part::Prism',f'LidNutPocket{i}');h.Polygon=6;h.Circumradius=C['lid_nut_af']/math.sqrt(3);h.Height=C['lid_nut_depth']+1;h.Placement.Base=App.Vector(x,y,55-C['lid_nut_depth']);construction.addObject(h);drills.append(h)
base=cut('Base',reinforced,multi('BaseDrills',drills));base.Label='01 Base - vented, board mounts, cable access'

lidblank=box('LidBlank',0,0,'Parameters.Height','Parameters.Width','Parameters.Depth','Parameters.Lid')
lidcuts=[]
for i,x in enumerate(range(22,107,7)):
    lidcuts.append(box(f'LidVent{i}',x,32,54,4,68,5))
lidcuts += [cyl(f'LidHole{i}',x,y,54,C['lid_screw_clearance']/2,5) for i,(x,y) in enumerate(C['lid_screw_centers'],1)]
lid=cut('Lid',lidblank,multi('LidCuts',lidcuts));lid.Label='02 Lid - 13 ventilation slots / 4 M3 screws'

# Drill the board reference at the actual selected holes; no fake components.
boardblank=box('BoardOutline',14,24,11,100,79,1.57)
boardtools=[cyl(f'RefBoardHole{i}',x,y,10,1.375,4) for i,(x,y) in enumerate(board_holes,1)]
board=cut('CarrierReference',boardblank,multi('ReferenceHoles',boardtools));board.Label='REF - P3768 board outline and four mounting holes'
top=box('TopClearance',14,24,12.57,100,79,35.86);top.Label='REF - conservative upper keepout, NOT exact component CAD'
underblank=box('BottomClearanceBlank',14,24,6.7,100,79,4.3)
underpads=[cyl(f'MountPadAllowance{i}',x,y,6,3,6) for i,(x,y) in enumerate(board_holes,1)]
under=cut('BottomClearance',underblank,multi('MountPadAllowances',underpads));under.Label='REF - underside keepout minus assumed 6 mm mount-pad zones; verify fit'
for o in [board,top,under]:refs.addObject(o)
construction.removeObject(base);construction.removeObject(lid)
doc.recompute()

report={'revision':'A0','units':'mm','objects':{},'checks':{},'limitations':[
 'Reference geometry is an envelope, not an exact Jetson component model.',
 'Carrier hole pattern sourced from NVIDIA A04 layout. Factory base removal and actual screw compatibility require physical fit check.',
 'Keepouts do not establish thermal performance, cable bend clearance, vibration strength, EMC or vehicle suitability.',
 'Changing major envelope parameters requires corresponding hole/vent updates and regeneration; not every layout feature follows spreadsheet edits.'
]}
for o in [base,lid,board]:
    s=o.Shape;b=s.BoundBox
    report['objects'][o.Name]={'valid':s.isValid(),'solids':len(s.Solids),'volume_mm3':s.Volume,'bounds_mm':[b.XLength,b.YLength,b.ZLength]}
    assert s.isValid() and len(s.Solids)==1,(o.Name,report['objects'][o.Name])
def no_overlap(label,a,b):
    volume=a.Shape.common(b.Shape).Volume;report['checks'][label]={'overlap_mm3':volume,'pass':volume<1e-6};assert volume<1e-6,(label,volume)
no_overlap('base_lid',base,lid)
no_overlap('base_board',base,board)
no_overlap('base_upper_keepout',base,top)
no_overlap('lid_upper_keepout',lid,top)
no_overlap('base_lower_keepout',base,under)
report['checks']['bottom_clearance_mm']=8-4.3
report['checks']['conservative_top_clearance_mm']=55-(11+1.57+35.86)
report['checks']['board_hole_centers_mm']=board_holes
report['checks']['board_hole_pitch_mm']=[86,58]
report['checks']['lid_open_area_mm2']=len(range(22,107,7))*4*68
report['checks']['lid_open_area_fraction']=report['checks']['lid_open_area_mm2']/(128*118)
doc.recompute()
doc.saveAs(str(ROOT/'sanash_enclosure.FCStd'))
Part.export([base,lid],str(ROOT/'sanash_enclosure.step'))
Part.export([base],str(ROOT/'sanash_base.step'))
Part.export([lid],str(ROOT/'sanash_lid.step'))
for obj,name in [(base,'sanash_base'),(lid,'sanash_lid')]:
    sh=obj.Shape.copy()
    if obj==lid:sh.translate(App.Vector(0,0,-55))
    mesh=MeshPart.meshFromShape(Shape=sh,LinearDeflection=.08,AngularDeflection=.15,Relative=False)
    mesh.write(str(ROOT/(name+'.stl')))
    assert mesh.isSolid(),name
    report['objects'][obj.Name]['stl_watertight']=mesh.isSolid()
# Re-import STEP independently to verify transferable solids.
step=Part.read(str(ROOT/'sanash_enclosure.step'))
assert step.isValid() and len(step.Solids)==2
report['checks']['step_reimport']={'valid':step.isValid(),'solids':len(step.Solids)}
(ROOT/'mechanical_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
