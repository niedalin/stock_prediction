"""Matplotlib 中文字体：优先系统字体（如 macOS PingFang SC），避免标题/图例乱码。"""
import matplotlib
import matplotlib.font_manager as fm


def configure_cjk_font():
    preferred = [
        'PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'Songti SC', 'STHeiti',
        'Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC',
    ]
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['axes.unicode_minus'] = False
    chosen = None
    for name in preferred:
        path = fm.findfont(fm.FontProperties(family=name))
        if path and 'dejavu' not in path.lower():
            chosen = name
            break
    if chosen:
        rest = [x for x in preferred if x != chosen]
        matplotlib.rcParams['font.sans-serif'] = [chosen] + rest + ['DejaVu Sans', 'sans-serif']
    else:
        matplotlib.rcParams['font.sans-serif'] = preferred + ['DejaVu Sans', 'sans-serif']


def cjk_font_properties():
    for name in matplotlib.rcParams['font.sans-serif']:
        if name in ('sans-serif', 'DejaVu Sans'):
            continue
        path = fm.findfont(fm.FontProperties(family=name))
        if path and 'dejavu' not in path.lower():
            return fm.FontProperties(fname=path)
    return None


configure_cjk_font()
