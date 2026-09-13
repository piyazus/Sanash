"""Native KiCad system interconnect, not a carrier-board redesign.

Port symbols deliberately use semantic contact identifiers rather than inventing
PCB terminal numbers. Ethernet is an intact purchased cable and a logical port;
its internal pairs/magnetics are not represented as one electrical conductor.
"""
import json
import uuid
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME = "sanash_gateway"
NS = uuid.UUID("984541c0-cd68-493c-b6d5-ebf1ab2764ac")
def uid(s): return str(uuid.uuid5(NS, s))
def q(s): return json.dumps(str(s), ensure_ascii=False)
def fx(size=1.27, extra=""): return f"(effects (font (size {size} {size})) {extra})"
def prop(k,v,x,y,hide=False):
    return f'(property {q(k)} {q(v)} (at {x} {y} 0) {fx(extra="(hide yes)" if hide else "")})'

# x,y coordinates here are local symbol coordinates (y upwards).
definitions = {
  "KitDC": dict(w=23,h=13,ref="A",value="JETSON / J16 DC INPUT",pins=[("CENTER","+DC",-28,5,0,"power_in"),("SLEEVE","RETURN",-28,-5,0,"power_in")]),
  "KitLAN": dict(w=23,h=10,ref="A",value="JETSON / J15 ETHERNET",pins=[("PORT","RJ45 port",-28,0,0,"passive")]),
  "KitPSU": dict(w=23,h=13,ref="PS",value="INCLUDED NVIDIA ADAPTER",pins=[("CENTER","+19V",28,5,180,"power_out"),("SLEEVE","RETURN",28,-5,180,"power_out")]),
  "VideoLAN": dict(w=23,h=10,ref="X",value="AUTHORISED VIDEO NETWORK",pins=[("PORT","RJ45 port",28,0,180,"passive")]),
}
def lib(n,d,qualified=True):
    pins="".join(f'(pin {typ} line (at {x} {y} {a}) (length 5) (name {q(label)} {fx(1.1)}) (number {q(num)} {fx(.9)}))' for num,label,x,y,a,typ in d['pins'])
    return f'''(symbol {q("SANASH:"+n if qualified else n)}
      (pin_names (offset 1)) (in_bom yes) (on_board no)
      {prop("Reference",d['ref'],0,d['h']+5)} {prop("Value",d['value'],0,d['h']+2)}
      {prop("Footprint","",0,0,True)} {prop("Datasheet","",0,0,True)}
      (symbol {q(n+"_0_1")} (rectangle (start {-d['w']} {d['h']}) (end {d['w']} {-d['h']}) (stroke (width .254) (type default)) (fill (type background))))
      (symbol {q(n+"_1_1")} {pins}))'''

sheet=uid('sheet')
items=[]
def place(n,ref,x,y):
    d=definitions[n]
    items.append(f'''(symbol (lib_id "SANASH:{n}") (at {x} {y} 0) (unit 1) (in_bom yes) (on_board no) (dnp no) (uuid "{uid(ref)}")
    {prop('Reference',ref,x,y-d['h']-6)} {prop('Value',d['value'],x,y-d['h']-2.5)}
    {prop('Footprint','',x,y,True)} {prop('Datasheet','',x,y,True)}
    {''.join(f'(pin {q(p[0])} (uuid "{uid(ref+p[0])}"))' for p in d['pins'])}
    (instances (project "{NAME}" (path "/{sheet}" (reference "{ref}") (unit 1)))))''')
def text(s,x,y,size=1.27):
    items.append(f'(text {q(s)} (at {x} {y} 0) {fx(size,"(justify left)")} (uuid "{uid(s+str(x)+str(y))}"))')
def wire(x1,y1,x2,y2):
    items.append(f'(wire (pts (xy {x1} {y1}) (xy {x2} {y2})) (stroke (width 0) (type default)) (uuid "{uid(str((x1,y1,x2,y2)))}"))')
def label(s,x,y):
    items.append(f'(label {q(s)} (at {x} {y} 0) {fx(1.1,"(justify left bottom)")} (uuid "{uid(s)}"))')

text("SANASH  /  BENCH GATEWAY INTERCONNECT",20,18,2.5)
text("A0 - Candidate: existing camera network -> Jetson -> occupancy output",20,25,1.5)
place("KitPSU","PS1",64,58)
place("KitDC","A1",222,58)
for yy,net in [(53,"DC_19V_CENTER"),(63,"DC_RETURN_SLEEVE")]:
    wire(92,yy,194,yy);label(net,116,yy)
text("W1  Factory adapter cable / barrel plug 5.5 x 2.5 x 9.5 mm",95,75,1.1)
text("Use the supplied adapter; centre positive. No custom power PCB in this revision.",20,83,1.2)
text("NVIDIA sec. 3.8: jack 3.5 A max; table 5-2 separately lists rail 4.2 A. Do not treat 4.2 A as jack rating.",20,89,1.1)
place("VideoLAN","X1",64,116)
place("KitLAN","A2",222,116)
wire(92,116,194,116);label("ETHERNET_CABLE_ASSEMBLY",115,116)
text("W2  Purchased Cat 6 patch cable, intact 8P8C terminations",96,132,1.1)
text("A1 and A2 are two external interfaces of ONE Jetson developer kit.",20,143,1.2)
text("Ethernet line denotes a complete cable/link, not one pin or a copper net to manufacture.",20,150,1.2)
text("X1 requires a confirmed authorised network stream. Recorder model, codec and stream access are OPEN.",20,157,1.1)
text("Existing cameras, recorder and driver's monitor retain their normal connections.",20,164,1.2)
text("STAND CONFIGURATION",20,176,1.4)
text("microSD storage; stock fan retained; wired bench LAN. No LTE modem, video decoder or vehicle DC input fitted.",20,183,1.1)
text("NVIDIA SP-11324-001 v1.3, secs. 2.2, 3.8 / Hardware Layout user guide. See README for source links.",20,190,1.0)

sch=f'''(kicad_sch (version 20250114) (generator "eeschema") (generator_version "10.0")
 (uuid "{sheet}") (paper "A4")
 (title_block (title "SANASH bench gateway - system interconnect") (date "2026-09-12") (rev "A0") (company "SANASH") (comment 1 "Not a vehicle power circuit or PCB fabrication schematic"))
 (lib_symbols {''.join(lib(n,d) for n,d in definitions.items())})
 {''.join(items)} (embedded_fonts no))'''
def grid(s):
    # Standard 50 mil connection grid. Snap symbol geometry and placement alike.
    def pair(m):
        return f'({m[1]} {round(round(float(m[2])/1.27)*1.27,4)} {round(round(float(m[3])/1.27)*1.27,4)}'
    s=re.sub(r'\((at|xy|start|end) (-?[\d.]+) (-?[\d.]+)',pair,s)
    return re.sub(r'\(length ([\d.]+)\)',lambda m:f'(length {round(round(float(m[1])/1.27)*1.27,4)})',s)
(ROOT/(NAME+'.kicad_sch')).write_text(grid(sch),encoding='utf-8')
(ROOT/'SANASH.kicad_sym').write_text(grid('(kicad_symbol_lib (version 20250114) (generator "kicad_symbol_editor") '+''.join(lib(n,d,False) for n,d in definitions.items())+')'),encoding='utf-8')
(ROOT/'sym-lib-table').write_text('(sym_lib_table (version 7) (lib (name "SANASH")(type "KiCad")(uri "${KIPRJMOD}/SANASH.kicad_sym")(options "")(descr "SANASH external module interfaces")))',encoding='utf-8')
(ROOT/(NAME+'.kicad_pro')).write_text(json.dumps({'meta':{'filename':NAME+'.kicad_pro','version':1}},indent=2))
print('Created native KiCad project, schematic and project-local symbol library')
