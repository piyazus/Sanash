"""Independent connectivity and vendor-coordinate checks, Python standard library."""
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
p=json.loads((R/'design_parameters.json').read_text())
erc=json.loads((R/'erc.json').read_text())
assert not [v for s in erc['sheets'] for v in s['violations']]
root=ET.parse(R/'interconnect.net.xml').getroot()
nets={net.attrib['name'].removeprefix('/'):{(x.attrib['ref'],x.attrib['pin']) for x in net.findall('node')} for net in root.findall('nets/net')}
expected={
 'DC_19V_CENTER':{('PS1','CENTER'),('A1','CENTER')},
 'DC_RETURN_SLEEVE':{('PS1','SLEEVE'),('A1','SLEEVE')},
 'ETHERNET_CABLE_ASSEMBLY':{('X1','PORT'),('A2','PORT')},
}
assert nets==expected,(nets,expected)
mechanical=json.loads((R/'mechanical_validation.json').read_text())
for check in mechanical['checks'].values():
    if isinstance(check,dict) and 'pass' in check:assert check['pass']
with (R/'bom.csv').open(newline='') as f:
    bom=list(csv.DictReader(f));assert all(None not in row and all(v is not None for v in row.values()) for row in bom)

src=R/'references/P3768_A04.brd.alg'
coords=set();evidence=[];header=[]
with src.open() as f:
    for number,line in enumerate(f,1):
        if line.startswith('A!'):header=line.rstrip().split('!')[1:-1]
        if line.startswith('S!'):
            row=dict(zip(header,line.rstrip().split('!')[1:-1]))
            if row.get('REFDES')=='MEC5' and row.get('PAD_STACK_NAME')=='NTH_275P' and row.get('PIN_X') and row.get('PIN_Y'):
                point=(float(row['PIN_X']),float(row['PIN_Y']))
                if point not in coords:
                    evidence.append({'source_line':number,'layout_xy_mm':point,'pad_stack':'NTH_275P','header':header,'record':line.rstrip()});coords.add(point)
assert set(map(tuple,p['nvidia_reference']['selected_layout_holes_xy']))<=coords
selected=p['nvidia_reference']['selected_layout_holes_xy']
computed=[[p['design_choices']['board_origin_x']+x+4,p['design_choices']['board_origin_y']+y+17] for x,y in selected]
assert computed==mechanical['checks']['board_hole_centers_mm']
source_files=['nvidia_carrier_spec_v1.3.pdf','nvidia_reference_a04.zip','P3768_A04.brd.alg']
provenance={
 'retrieved_on':'2026-09-12',
 'spec_url':'https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf',
 'layout_url':'https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_reference_design_files_a04_20230320.zip/',
 'source_hashes':{name:hashlib.sha256((R/'references'/name).read_bytes()).hexdigest() for name in source_files},
 'outline_mm':[-4,-17,96,62],
 'transformation':'Board local = layout + (4,17); enclosure = board local + (14,24). No pixel-derived hole positions.',
 'hole_evidence':evidence,
 'selected_holes':selected,
 'other_holes_not_used':[[x,y] for x,y in sorted(coords) if [x,y] not in selected]
}
(R/'references/source_provenance.json').write_text(json.dumps(provenance,indent=2))
result={'erc_violations':0,'expected_nets_verified':3,'vendor_hole_coordinates_verified':4,'bom_rows_verified':len(bom),'mechanical_report':'mechanical_validation.json','native_render_status':(R/'render_status.txt').read_text()}
assert result['native_render_status'].startswith('OK:')
(R/'package_validation.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
