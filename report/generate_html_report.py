import os
import base64
from io import BytesIO


def _fig_to_base64(fig):
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to convert figure: {e}")
        return None


def generate_html_report(metrics, plots, output_path="evaluation_report.html"):
    """
    Generate a simple HTML evaluation report with metrics and plots.
    Windows-safe, robust version.
    """

    print("📝 Generating HTML report...")

    # ---- Ensure directory exists (Windows-safe) ----
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # ---- Convert plots safely ----
    plot_imgs = {}
    if plots:
        for name, fig in plots.items():
            img = _fig_to_base64(fig)
            if img:
                plot_imgs[name] = img

    # ---- Metrics table ----
    metrics_rows = ""
    for k, v in metrics.items():
        try:
            val = float(v)
            metrics_rows += f"<tr><td>{k}</td><td>{val:.5f}</td></tr>"
        except Exception:
            metrics_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    # ---- Plots section ----
    plots_html = ""
    if plot_imgs:
        for name, img in plot_imgs.items():
            plots_html += f"""
            <div class="plot">
                <h3>{name}</h3>
                <img src="data:image/png;base64,{img}" />
            </div>
            """
    else:
        plots_html = "<p>No plots available</p>"

    # ---- HTML template ----
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>ML Evaluation Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background: #f5f5f5;
            }}
            h1 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 50%;
                background: white;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            .plot {{
                margin-top: 30px;
            }}
            img {{
                max-width: 600px;
                border: 1px solid #ccc;
                background: white;
            }}
        </style>
    </head>

    <body>
        <h1>📊 Model Evaluation Report</h1>

        <h2>Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>

        <h2>Plots</h2>
        {plots_html}

    </body>
    </html>
    """

    # ---- Save (Windows-safe encoding) ----
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Report saved to {output_path}")