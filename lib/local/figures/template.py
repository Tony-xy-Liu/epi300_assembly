from plotly import subplots as sp, graph_objs as go

def BaseFigure(shape: tuple[int, int]=(1, 1), **kwargs) -> go.Figure:
    ncols, nrows = shape
    # column_widths=[0.1, 0.9], row_heights=[0.3, 0.1, 0.6],
    params: dict = dict(
        rows=nrows, cols=ncols,
        horizontal_spacing=0.02, vertical_spacing=0.02,
        shared_yaxes=True, shared_xaxes=True,
    ) | kwargs
    return sp.make_subplots(**params)

def ApplyTemplate(fig: go.Figure, default_xaxis: dict = dict(), default_yaxis: dict = dict(), axis: dict[str, dict] = dict(), layout: dict = dict()):
    # @axis
    # example: {"1 1 y": dict(showticklabels=True, categoryorder='array', categoryarray=cat_list)}
    # params: https://plotly.com/python/reference/layout/xaxis/

    color_none = 'rgba(0,0,0,0)'
    color_axis = 'rgba(0, 0, 0, 0.15)'
    axis_template = dict(showgrid=False, showticklabels=True, linecolor="black", linewidth=1, ticks="outside", gridcolor=color_axis, zerolinecolor=color_none, zerolinewidth=1)
    DEF_XAXIS: dict = axis_template|default_xaxis
    DEF_YAXIS: dict = axis_template|default_yaxis
    logged_cols, logged_rows = [], []
    _layout = layout.copy()
    _rows, _ncols = fig._get_subplot_rows_columns()
    nrows, ncols = [len(x) for x in [_rows, _ncols]]
    for i in range(nrows*ncols):
        x, y = i%ncols + 1, i//ncols + 1
        i += 1
        ax = DEF_XAXIS | axis.get(f"{x} {y} x", DEF_XAXIS.copy())
        ay = DEF_YAXIS | axis.get(f"{x} {y} y", DEF_YAXIS.copy())
        if x in logged_cols: ax |= dict(type="log")
        if y in logged_rows: ay |= dict(type="log")
        _layout[f"xaxis{i if i != 1 else ''}"] = ax
        _layout[f"yaxis{i if i != 1 else ''}"] = ay
    
    bg_col="white"
    W, H = 1000, 600
    _layout: dict = dict(
        width=W, height=H,
        paper_bgcolor=bg_col,
        plot_bgcolor=bg_col,
        margin=dict(
            l=15, r=15, b=15, t=15, pad=5
        ),
        font=dict(
            family="Arial",
            size=16,
            color="black",
        ),
        legend=dict(
            font=dict(
                size=12,
            ),
        ),
    ) | _layout
    fig.update_layout(**_layout)
    return fig
