"""SentinelML design system — "Mission Control".

Deep-space HUD aesthetic: starfield + radar-sweep hero (canvas, animated in an
iframe), HUD corner-bracket panels, rotating conic-gradient borders, scanline
overlay and monospace telemetry readouts.
"""
from __future__ import annotations

import html
from collections.abc import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

# ----------------------------------------------------------------- palette
BRAND = "SentinelML"
TAGLINE = "Mission control for model reliability"
PRIMARY = "#7c83ff"     # signal indigo
SECONDARY = "#2ee6d6"   # telemetry cyan
ACCENT = "#c084fc"      # nebula violet
SUCCESS, WARNING, DANGER, MUTED = "#34d399", "#fbbf24", "#fb7185", "#8b93b8"
BG, SURFACE, BORDER, TEXT = "#04061a", "rgba(124,131,255,0.06)", "rgba(124,131,255,0.22)", "#e6e9ff"
COLORWAY = [PRIMARY, SECONDARY, ACCENT, SUCCESS, WARNING, DANGER, "#f472b6", "#a3e635"]
POLICY_COLORS = {"static": MUTED, "periodic": WARNING, "drift_aware": SECONDARY}
SEVERITY_COLORS = {"critical": DANGER, "high": DANGER, "medium": WARNING, "low": SECONDARY, "info": PRIMARY}

pio.templates["sentinel"] = go.layout.Template(layout=dict(
    font=dict(family="'IBM Plex Mono', monospace", color=TEXT, size=12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(124,131,255,0.03)", colorway=COLORWAY,
    xaxis=dict(gridcolor="rgba(124,131,255,0.12)", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor="rgba(124,131,255,0.12)", zeroline=False, linecolor=BORDER, tickfont=dict(color=MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1, x=0, font=dict(color=MUTED)),
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor="#0b0f2e", bordercolor=SECONDARY, font=dict(family="IBM Plex Mono", color=TEXT)),
    transition=dict(duration=600, easing="cubic-in-out"),
))
pio.templates.default = "plotly_dark+sentinel"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
@property --ang { syntax: '<angle>'; inherits: false; initial-value: 0deg; }
@property --ring { syntax: '<number>'; inherits: false; initial-value: 0; }

html, body, [class*="css"], .stMarkdown { font-family: 'Space Grotesk', system-ui, sans-serif; }
code, .mono { font-family: 'IBM Plex Mono', monospace !important; }

/* deep-space background with drifting stars (pure CSS) */
.stApp {
  background:
    radial-gradient(1px 1px at 20% 30%, #fff8 50%, transparent 51%),
    radial-gradient(1px 1px at 70% 80%, #fff6 50%, transparent 51%),
    radial-gradient(1.5px 1.5px at 40% 60%, #c7d2fe 50%, transparent 51%),
    radial-gradient(1px 1px at 85% 20%, #fff7 50%, transparent 51%),
    radial-gradient(900px 600px at 0% 0%, #1b1f6b55, transparent 60%),
    radial-gradient(800px 600px at 100% 100%, #0e5c6655, transparent 60%),
    __BG__;
  background-size: 300px 300px, 400px 400px, 500px 500px, 350px 350px, auto, auto, auto;
  animation: starDrift 120s linear infinite;
}
@keyframes starDrift { to { background-position: 300px 600px, -400px 400px, 500px -500px, -350px 700px, 0 0, 0 0, 0 0; } }
/* moving scanline */
.stApp::after { content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 999;
  background: linear-gradient(transparent 0%, #2ee6d60c 50%, transparent 100%); height: 140px;
  animation: scan 9s linear infinite; }
@keyframes scan { from { transform: translateY(-140px); } to { transform: translateY(110vh); } }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #070a26f2, #04061af2); border-right: 1px solid __BORDER__; }
[data-testid="stSidebarNav"] a, [data-testid="stSidebarNavLink"] { border-radius: 0 !important; border-left: 2px solid transparent; transition: all .25s; font-family: 'IBM Plex Mono', monospace; letter-spacing: .02em; }
[data-testid="stSidebarNavLink"]:hover { border-left-color: __SECONDARY__; background: #2ee6d612 !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { border-left-color: __SECONDARY__; background: linear-gradient(90deg, #2ee6d622, transparent) !important; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1440px; }

@keyframes rise { from { opacity: 0; transform: translateY(16px) scale(.985); filter: blur(6px); } to { opacity: 1; transform: none; filter: none; } }
@keyframes slide { from { opacity: 0; transform: translateX(-18px); } to { opacity: 1; transform: none; } }
@keyframes spin { to { --ang: 360deg; } }
@keyframes ringFill { from { --ring: 0; } }
@keyframes blip { 0%,100% { opacity: .35; } 50% { opacity: 1; } }
@keyframes flow { to { stroke-dashoffset: -28; } }
.element-container, .stPlotlyChart { animation: rise .7s cubic-bezier(.16,1,.3,1) both; }

/* HUD panel: corner brackets */
.hud { position: relative; padding: 18px 20px; background: __SURFACE__; border: 1px solid #7c83ff22;
  clip-path: polygon(0 12px, 12px 0, 100% 0, 100% calc(100% - 12px), calc(100% - 12px) 100%, 0 100%); }
.hud::before, .hud::after { content: ""; position: absolute; width: 18px; height: 18px; border: 2px solid __SECONDARY__; }
.hud::before { top: 6px; right: 6px; border-left: 0; border-bottom: 0; }
.hud::after { bottom: 6px; left: 6px; border-right: 0; border-top: 0; }

/* KPI tiles with rotating conic border */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin: 8px 0 20px; }
.kpi { position: relative; padding: 2px; border-radius: 16px; animation: rise .7s cubic-bezier(.16,1,.3,1) both;
  background: conic-gradient(from var(--ang), transparent 0 55%, var(--c1) 75%, __SECONDARY__ 90%, transparent 100%);
  animation: rise .7s cubic-bezier(.16,1,.3,1) both, spin 5s linear infinite; }
.kpi > .in { border-radius: 14px; box-shadow: inset 0 0 0 1px #7c83ff22; padding: 16px 18px; background: linear-gradient(160deg, #0b1036, #060824); height: 100%;
  transition: transform .3s; }
.kpi:hover > .in { transform: scale(.985); }
.kpi .label { font-family: 'IBM Plex Mono', ui-monospace, monospace; color: __MUTED__; font-size: .7rem; letter-spacing: .14em; text-transform: uppercase; }
.kpi .label::before { content: "▸ "; color: var(--c1); }
.kpi .value { font-size: 1.85rem; font-weight: 700; color: #fff; margin-top: 4px; letter-spacing: -.02em;
  text-shadow: 0 0 18px color-mix(in srgb, var(--c1) 55%, transparent); }
.kpi .delta { font-family: 'IBM Plex Mono', ui-monospace, monospace; font-size: .75rem; margin-top: 2px; }
.kpi .icon { position: absolute; right: 14px; top: 12px; font-size: 1.2rem; opacity: .6; animation: blip 3s ease-in-out infinite; }

.section { display: flex; align-items: baseline; gap: 12px; margin: 28px 0 10px; animation: slide .5s ease both; }
.section .idx { font-family: 'IBM Plex Mono', ui-monospace, monospace; color: __SECONDARY__; font-size: .78rem; }
.section h3 { margin: 0; font-size: 1.1rem; font-weight: 600; color: #fff; letter-spacing: .01em; }
.section .sub { color: __MUTED__; font-size: .8rem; font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.section .rule { flex: 1; height: 1px; background: linear-gradient(90deg, __BORDER__, transparent); position: relative; overflow: hidden; }
.section .rule::after { content: ""; position: absolute; inset: 0; width: 30%; background: linear-gradient(90deg, transparent, __SECONDARY__, transparent);
  animation: sweep 3.5s ease-in-out infinite; }
@keyframes sweep { from { transform: translateX(-100%); } to { transform: translateX(400%); } }

.card { margin-bottom: 12px; animation: rise .7s ease both; }
.card h4 { margin: 0 0 6px; color: #fff; font-size: .98rem; }
.card p { margin: 0; color: __MUTED__; font-size: .88rem; line-height: 1.55; }

.tl-item { position: relative; padding: 10px 14px 10px 30px; margin-bottom: 8px; border: 1px solid #7c83ff1f;
  background: #0a0e30aa; border-radius: 4px; animation: slide .45s ease both; }
.tl-item::before { content: ""; position: absolute; left: 11px; top: 15px; width: 8px; height: 8px; border-radius: 50%;
  background: var(--sev); box-shadow: 0 0 0 0 var(--sev); animation: ping 2s infinite; }
@keyframes ping { 0% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--sev) 70%, transparent); } 80%,100% { box-shadow: 0 0 0 9px transparent; } }
.tl-item .meta { font-size: .72rem; color: __MUTED__; font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.tl-item .msg { font-size: .88rem; color: __TEXT__; margin-top: 3px; }
.chip { display: inline-block; padding: 1px 8px; font-size: .68rem; font-weight: 600; font-family: 'IBM Plex Mono', ui-monospace, monospace;
  text-transform: uppercase; letter-spacing: .08em; color: var(--c); border: 1px solid var(--c); background: color-mix(in srgb, var(--c) 12%, transparent); }

.ring { width: 136px; height: 136px; border-radius: 50%; display: grid; place-items: center; margin: auto; position: relative;
  background: conic-gradient(var(--rc) calc(var(--ring) * 1%), #7c83ff1a 0); animation: ringFill 1.8s cubic-bezier(.16,1,.3,1) both; --ring: var(--val); }
.ring::after { content: ""; position: absolute; inset: -6px; border-radius: 50%; border: 1px dashed #7c83ff55; animation: rot 18s linear infinite; }
@keyframes rot { to { transform: rotate(360deg); } }
.ring > div { width: 108px; height: 108px; border-radius: 50%; background: __BG__; display: grid; place-items: center;
  font: 700 1.5rem 'IBM Plex Mono', ui-monospace, monospace; color: #fff; }

.flow { display: flex; align-items: center; flex-wrap: wrap; margin: 6px 0 12px; }
.flow .node { padding: 10px 14px; border: 1px solid __BORDER__; background: #0a0e30; font-weight: 600; font-size: .84rem; color: #fff;
  animation: rise .6s ease both; clip-path: polygon(8px 0, 100% 0, calc(100% - 8px) 100%, 0 100%); }
.flow .node small { display: block; color: __MUTED__; font: 500 .68rem 'IBM Plex Mono', ui-monospace, monospace; }
.flow svg { width: 40px; height: 14px; }
.flow svg line { stroke: __SECONDARY__; stroke-width: 2; stroke-dasharray: 4 10; stroke-linecap: round; animation: flow .8s linear infinite; }

.stButton > button, .stDownloadButton > button { border-radius: 0; border: 1px solid __SECONDARY__; background: #2ee6d614; color: __SECONDARY__;
  font-family: 'IBM Plex Mono', ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; font-size: .8rem; transition: all .2s;
  clip-path: polygon(8px 0, 100% 0, 100% calc(100% - 8px), calc(100% - 8px) 100%, 0 100%, 0 8px); }
.stButton > button:hover { background: __SECONDARY__; color: #04061a; box-shadow: 0 0 24px #2ee6d688; }
[data-testid="stDataFrame"] { border: 1px solid __BORDER__; }
.stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.disclaimer { font: .76rem 'IBM Plex Mono', ui-monospace, monospace; color: __MUTED__; border-left: 2px solid __WARNING__; padding: 8px 12px; margin-top: 18px; background: #fbbf240a; }
.footer { text-align: center; color: __MUTED__; font: .72rem 'IBM Plex Mono', ui-monospace, monospace; margin-top: 40px; letter-spacing: .1em; }
.brand { font: 700 1.25rem 'Space Grotesk', system-ui, sans-serif; color: #fff; letter-spacing: -.01em; }
.brand small { display: block; font: 500 .7rem 'IBM Plex Mono', ui-monospace, monospace; color: __SECONDARY__; letter-spacing: .12em; text-transform: uppercase; margin-top: 2px; }
.status { font: .72rem 'IBM Plex Mono', ui-monospace, monospace; color: __MUTED__; display: flex; gap: 8px; align-items: center; }
.status .led { width: 7px; height: 7px; border-radius: 50%; background: __SUCCESS__; box-shadow: 0 0 10px __SUCCESS__; animation: blip 1.4s infinite; }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
</style>
"""

_TOKENS = {"__BG__": BG, "__SURFACE__": SURFACE, "__BORDER__": BORDER, "__TEXT__": TEXT, "__MUTED__": MUTED,
           "__PRIMARY__": PRIMARY, "__SECONDARY__": SECONDARY, "__ACCENT__": ACCENT, "__SUCCESS__": SUCCESS,
           "__WARNING__": WARNING, "__DANGER__": DANGER}


def _fill(s: str) -> str:
    for k, v in _TOKENS.items():
        s = s.replace(k, v)
    return s


def _e(s: object) -> str:
    return html.escape(str(s))


def apply_theme() -> None:
    st.markdown(_fill(_CSS), unsafe_allow_html=True)


_HERO = """
<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body{margin:0;background:transparent;overflow:hidden;font-family:'Space Grotesk',sans-serif;color:#e6e9ff}
.wrap{position:relative;height:__H__px;border:1px solid #7c83ff33;border-radius:18px;overflow:hidden;
 background:radial-gradient(600px 300px at 80% 50%,#1b1f6b66,transparent 70%),linear-gradient(120deg,#060a2a,#04061a)}
canvas{position:absolute;inset:0;width:100%;height:100%}
.txt{position:absolute;left:34px;top:28px;right:340px}
.badge{display:inline-flex;gap:8px;align-items:center;font:500 11px 'IBM Plex Mono', ui-monospace, monospace;letter-spacing:.16em;text-transform:uppercase;
 color:#2ee6d6;border:1px solid #2ee6d655;padding:5px 10px;background:#2ee6d60f}
.led{width:7px;height:7px;border-radius:50%;background:#34d399;box-shadow:0 0 10px #34d399;animation:b 1.2s infinite}
@keyframes b{50%{opacity:.25}}
h1{margin:14px 0 8px;font-size:40px;letter-spacing:-.03em;font-weight:700}
h1 span{display:inline-block;opacity:0;transform:translateY(18px);animation:in .6s cubic-bezier(.16,1,.3,1) forwards}
@keyframes in{to{opacity:1;transform:none}}
p{margin:0;color:#8b93b8;font-size:14.5px;line-height:1.55;max-width:760px;min-height:44px}
p::after{content:'▍';color:#2ee6d6;animation:b 1s steps(1) infinite}
.tele{position:absolute;left:34px;bottom:18px;font:500 11px 'IBM Plex Mono', ui-monospace, monospace;color:#8b93b8;letter-spacing:.08em}
.tele b{color:#2ee6d6;font-weight:500}
@media(max-width:820px){.txt{right:24px}h1{font-size:30px}}
</style></head><body><div class="wrap"><canvas id="c"></canvas>
<div class="txt"><span class="badge"><span class="led"></span>__BADGE__</span><h1 id="t">__ICON__ __TITLE__</h1><p id="s"></p></div>
<div class="tele">SYS <b>NOMINAL</b> · UPLINK <b id="u">--</b> · T+<b id="clk">00:00:00</b></div></div>
<script>
const T=document.getElementById('t'),txt=T.textContent;T.innerHTML='';
[...txt].forEach((ch,i)=>{const s=document.createElement('span');s.textContent=ch===' '?'\\u00a0':ch;s.style.animationDelay=(i*0.035)+'s';T.appendChild(s)});
const sub=__SUB__,S=document.getElementById('s');let k=0;(function ty(){S.textContent=sub.slice(0,k);if(k++<sub.length)setTimeout(ty,12)})();
const c=document.getElementById('c'),x=c.getContext('2d');let W,H,stars=[];
function rs(){W=c.width=c.offsetWidth*devicePixelRatio;H=c.height=c.offsetHeight*devicePixelRatio;
 stars=Array.from({length:160},()=>({x:Math.random()*W,y:Math.random()*H,z:Math.random()*2+.3}))}
rs();addEventListener('resize',rs);
const blips=Array.from({length:7},()=>({a:Math.random()*6.28,r:.25+Math.random()*.7,life:0}));
let t0=performance.now();
function draw(t){x.clearRect(0,0,W,H);
 for(const s of stars){s.x-=s.z*.35;if(s.x<0)s.x=W;x.fillStyle=`rgba(199,210,254,${.25+s.z*.3})`;x.fillRect(s.x,s.y,s.z,s.z)}
 const R=Math.min(H*.42,150*devicePixelRatio),cx=W-R-40*devicePixelRatio,cy=H/2;
 x.strokeStyle='rgba(46,230,214,.22)';x.lineWidth=1;
 for(let i=1;i<=4;i++){x.beginPath();x.arc(cx,cy,R*i/4,0,6.28);x.stroke()}
 x.beginPath();x.moveTo(cx-R,cy);x.lineTo(cx+R,cy);x.moveTo(cx,cy-R);x.lineTo(cx,cy+R);x.stroke();
 const ang=(t/1000*1.4)%6.283;
 const g=x.createConicGradient?x.createConicGradient(ang-1.2,cx,cy):null;
 if(g){g.addColorStop(0,'rgba(46,230,214,0)');g.addColorStop(.19,'rgba(46,230,214,.35)');g.addColorStop(.191,'rgba(46,230,214,0)');
  x.fillStyle=g;x.beginPath();x.arc(cx,cy,R,0,6.28);x.fill()}
 x.strokeStyle='#2ee6d6';x.lineWidth=2*devicePixelRatio;x.beginPath();x.moveTo(cx,cy);x.lineTo(cx+Math.cos(ang)*R,cy+Math.sin(ang)*R);x.stroke();
 for(const b of blips){let d=((ang-b.a)%6.283+6.283)%6.283;if(d<.06)b.life=1;b.life*=.985;
  if(b.life>.02){const bx=cx+Math.cos(b.a)*R*b.r,by=cy+Math.sin(b.a)*R*b.r;
   x.fillStyle=`rgba(251,113,133,${b.life})`;x.beginPath();x.arc(bx,by,4*devicePixelRatio,0,6.28);x.fill();
   x.strokeStyle=`rgba(251,113,133,${b.life*.6})`;x.beginPath();x.arc(bx,by,(4+(1-b.life)*16)*devicePixelRatio,0,6.28);x.stroke()}}
 const e=Math.floor((t-t0)/1000),hh=String(Math.floor(e/3600)).padStart(2,'0'),mm=String(Math.floor(e/60)%60).padStart(2,'0'),ss=String(e%60).padStart(2,'0');
 document.getElementById('clk').textContent=hh+':'+mm+':'+ss;document.getElementById('u').textContent=(98+Math.sin(t/900)*1.5).toFixed(1)+'%';
 requestAnimationFrame(draw)}
requestAnimationFrame(draw);
</script></body></html>
"""


def hero(title: str, subtitle: str, badge: str = "Live", icon: str = "🛰️", height: int = 236) -> None:
    import json

    doc = (_HERO.replace("__H__", str(height - 6)).replace("__BADGE__", _e(badge)).replace("__ICON__", _e(icon))
           .replace("__TITLE__", _e(title)).replace("__SUB__", json.dumps(subtitle)))
    components.html(doc, height=height)


def kpis(items: Iterable[tuple]) -> None:
    cards = []
    for i, it in enumerate(items):
        label, value, delta, icon, color = (list(it) + [None] * 5)[:5]
        color = color or PRIMARY
        d = ""
        if delta:
            dc = SUCCESS if str(delta).startswith(("+", "▲")) else DANGER if str(delta).startswith(("-", "−", "▼")) else MUTED
            d = f'<div class="delta" style="color:{dc}">{_e(delta)}</div>'
        cards.append(f'<div class="kpi" style="--c1:{color};animation-delay:{i * .09:.2f}s,0s"><div class="in">'
                     f'<div class="icon">{icon or ""}</div><div class="label">{_e(label)}</div>'
                     f'<div class="value">{_e(value)}</div>{d}</div></div>')
    st.markdown(f'<div class="kpi-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


_SECTION_IDX = {"n": 0}


def section(title: str, sub: str = "") -> None:
    _SECTION_IDX["n"] = _SECTION_IDX["n"] % 99 + 1
    st.markdown(f'<div class="section"><span class="idx">[{_SECTION_IDX["n"]:02d}]</span><h3>{_e(title)}</h3>'
                f'<span class="sub">{_e(sub)}</span><span class="rule"></span></div>', unsafe_allow_html=True)


def card(title: str, body: str, accent: str = PRIMARY) -> None:
    st.markdown(f'<div class="hud card" style="border-top:2px solid {accent}"><h4>{_e(title)}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def chip(text: str, color: str) -> str:
    return f'<span class="chip" style="--c:{color}">{_e(text)}</span>'


def timeline(events: list[dict], limit: int = 8) -> None:
    out = []
    for i, ev in enumerate(events[:limit]):
        sev = SEVERITY_COLORS.get(ev.get("severity", "info"), PRIMARY)
        out.append(f'<div class="tl-item" style="--sev:{sev};animation-delay:{i * .06:.2f}s">'
                   f'<div class="meta">{chip(ev.get("event_type", ""), sev)} &nbsp;B{_e(ev.get("batch_id", "–"))} · {_e(ev.get("ts", ""))}</div>'
                   f'<div class="msg">{_e(ev.get("message", ""))}</div></div>')
    st.markdown("".join(out) or '<div class="hud card"><p>No events yet.</p></div>', unsafe_allow_html=True)


def ring(value: float, label: str, color: str | None = None) -> None:
    color = color or (SUCCESS if value >= 80 else WARNING if value >= 60 else DANGER)
    st.markdown(f'<div style="text-align:center"><div class="ring" style="--val:{value:.0f};--rc:{color}"><div>{value:.0f}</div></div>'
                f'<div style="color:{MUTED};font:500 .72rem ui-monospace,monospace;letter-spacing:.1em;text-transform:uppercase;margin-top:10px">'
                f'{_e(label)}</div></div>', unsafe_allow_html=True)


def flow(nodes: list[tuple[str, str]]) -> None:
    arrow = '<svg viewBox="0 0 40 14"><line x1="2" y1="7" x2="38" y2="7"/></svg>'
    parts = [f'<div class="node" style="animation-delay:{i * .1:.1f}s">{_e(n)}<small>{_e(s)}</small></div>' for i, (n, s) in enumerate(nodes)]
    st.markdown(f'<div class="flow">{arrow.join(parts)}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(f'<div class="brand">🛰️ {BRAND}<small>{TAGLINE}</small></div>', unsafe_allow_html=True)


def status_line(text: str) -> None:
    st.markdown(f'<div class="status"><span class="led"></span>{_e(text)}</div>', unsafe_allow_html=True)


def footer() -> None:
    st.markdown(f'<div class="footer">{BRAND.upper()} // RESEARCH PROTOTYPE // FASTAPI · SCIKIT-LEARN · SQLITE · STREAMLIT</div>',
                unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=13, color=MUTED)) if title else None)
    return fig
