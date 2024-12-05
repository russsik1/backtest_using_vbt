import bokeh
from bokeh.plotting import figure, show
from bokeh.models import Legend, HoverTool, ColumnDataSource
from bokeh.palettes import Category20
import pandas as pd
from math import pi


def plot(rdf):
    d = rdf.copy()
    d['datetime'] = pd.to_datetime(d['open_date_time'] / 1000, unit='s', utc=True)
    d['date'] = d['datetime'].dt.strftime('%d.%m.%y %H:%M')
    w = 1  # 126060*1000 # half day in ms
    RED = Category20[7][6]
    GREEN = Category20[5][4]
    TOOLS = 'pan,wheel_zoom,box_zoom,reset,save'  # ,hover'
    source = ColumnDataSource(d)
    float_format = '0' * min(8, len(str(d['close'].values[0]).split('.')[1]))
    TOOLTIPS = [
        ("date", "@date"),
        ("open", "@open{0.%s}" % float_format),
        ("close", "@close{0.%s}" % float_format),
    ]
    df_green = d.loc[d['close'] > d['open']]
    df_red = d.loc[d['close'] < d['open']]
    source = ColumnDataSource(d)
    source_red = ColumnDataSource(df_red)
    source_green = ColumnDataSource(df_green)
    plot = figure(x_axis_type='linear', tools=TOOLS, tooltips=TOOLTIPS, width=1000, title='Candlestick')
    plot.sizing_mode = 'scale_both'
    plot.xaxis.major_label_orientation = pi / 4
    plot.grid.grid_line_alpha = 0.3
    plot.segment(d.index, d.high, d.index, d.low, color='black')
    plot.vbar('index', w, 'open', 'close', source=source_green, fill_color=GREEN, color="#D5E1DD",
              line_color='black')
    plot.vbar('index', w, 'open', 'close', source=source_red, fill_color=RED, color="#F2583E", line_color='black')
    plot.xaxis.major_label_overrides = {i: date.strftime('%Y-%m-%d %H:%M') for i, date in enumerate(d.datetime)}
    show(plot)