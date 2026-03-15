def render_plot(fig, container=None):
    """Helper to render Plotly figures responsively."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(250, 128, 114, 0.04)",
        plot_bgcolor="rgba(250, 128, 114, 0.02)",
        font=dict(color="#343A40", family="Poppins, sans-serif"),
        title=dict(font=dict(color=SALMON, size=20), x=0.5, xanchor="center", pad=dict(b=12)),
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0)", font=dict(color="#343A40")),
        colorway=[
            "rgba(255,245,240,0.55)",
            "rgba(254,224,210,0.55)",
            "rgba(252,187,161,0.55)",
            "rgba(252,146,114,0.55)",
            "rgba(251,106,74,0.55)",
            "rgba(239,59,44,0.55)"
        ],
    )
    fig.update_coloraxes(colorscale=GRADIENT_SCALE)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(201, 74, 68, 0.15)", zeroline=False, linecolor="rgba(201, 74, 68, 0.3)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(201, 74, 68, 0.15)", zeroline=False, linecolor="rgba(201, 74, 68, 0.3)")
    target = container if container is not None else st
    target.plotly_chart(fig, config={"responsive": True})
