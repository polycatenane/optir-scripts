# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.11.1", "numpy>=2.0", "pandas>=2.3"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # FOV comparison viewer
    """)
    return


@app.cell
def _():
    from pathlib import Path

    from hyperspectral_cells.fov_comparison import (
        comparison_figures,
        figure_control_selection,
        load_fov_comparisons,
        status_table,
    )
    import marimo as mo

    DATA_ROOT = Path("/var/home/praxis/Documents/Bai/Data/20260828 Artificial Sweetener") / "2026-08-25 and 2026-08-26"
    DEFAULT_GROUPS = ("0 hour", "Control", "Acesulfame", "Aspartame", "Carbon 13", "Saccharin", "Sucralose")
    return (
        DATA_ROOT,
        DEFAULT_GROUPS,
        Path,
        comparison_figures,
        figure_control_selection,
        load_fov_comparisons,
        mo,
        status_table,
    )


@app.cell
def _(DATA_ROOT, DEFAULT_GROUPS, mo):
    data_root = mo.ui.text(value=str(DATA_ROOT), label="Data root")
    groups = mo.ui.multiselect(
        options=list(DEFAULT_GROUPS), value=list(DEFAULT_GROUPS), label="Experimental groups"
    )
    wavenumber_1 = mo.ui.number(
        start=1000, stop=3000, step=1, value=1610, label="Wavenumber 1 (cm-1)"
    )
    wavenumber_2 = mo.ui.number(
        start=1000, stop=3000, step=1, value=1655, label="Wavenumber 2 (cm-1)"
    )
    load_form = mo.ui.dictionary(
        {
            "data_root": data_root,
            "groups": groups,
            "wavenumber_1": wavenumber_1,
            "wavenumber_2": wavenumber_2,
        }
    ).form(submit_button_label="Load FOV images")
    mo.vstack([load_form])
    return (load_form,)


@app.cell
def _(Path, load_form, load_fov_comparisons):
    submitted_settings = None if load_form.value is None else {
            "data_root": Path(load_form.value["data_root"]),
            "groups": tuple(load_form.value["groups"]),
            "wavenumber_1": float(load_form.value["wavenumber_1"]),
            "wavenumber_2": float(load_form.value["wavenumber_2"]),
        }
    rows = (
        load_fov_comparisons(
            submitted_settings["data_root"],
            submitted_settings["groups"],
            submitted_settings["wavenumber_1"],
            submitted_settings["wavenumber_2"],
        )
        if submitted_settings is not None
        else []
    )
    return rows, submitted_settings


@app.cell
def _(mo):
    selected_rows_per_figure, set_selected_rows_per_figure = mo.state(
        None, allow_self_loops=True
    )
    selected_time, set_selected_time = mo.state(None, allow_self_loops=True)
    return (
        selected_rows_per_figure,
        selected_time,
        set_selected_rows_per_figure,
        set_selected_time,
    )


@app.cell
def _(
    figure_control_selection,
    mo,
    rows,
    selected_rows_per_figure,
    selected_time,
    set_selected_rows_per_figure,
    set_selected_time,
):
    times, current_time, row_count_options, current_rows_per_figure = (
        figure_control_selection(rows, selected_time(), selected_rows_per_figure())
    )
    if rows and current_time != selected_time():
        set_selected_time(current_time)
    if rows and current_rows_per_figure != selected_rows_per_figure():
        set_selected_rows_per_figure(current_rows_per_figure)

    time_position = times.index(current_time) if current_time else -1
    row_count_position = row_count_options.index(current_rows_per_figure) if current_rows_per_figure else -1
    time_dropdown = mo.ui.dropdown(
        options=times,
        value=current_time,
        label="Hour",
        on_change=set_selected_time,
        disabled=not times,
    )
    rows_per_figure_dropdown = mo.ui.dropdown(
        options=row_count_options,
        value=current_rows_per_figure,
        label="Rows per figure",
        on_change=set_selected_rows_per_figure,
        disabled=not row_count_options,
    )
    previous_time = mo.ui.button(
        label="←",
        tooltip="Previous time",
        disabled=time_position <= 0,
        on_click=lambda _: set_selected_time(times[time_position - 1]),
    )
    next_time = mo.ui.button(
        label="→",
        tooltip="Next time",
        disabled=not times or time_position == len(times) - 1,
        on_click=lambda _: set_selected_time(times[time_position + 1]),
    )
    previous_rows_per_figure = mo.ui.button(
        label="←",
        tooltip="Fewer rows per figure",
        disabled=row_count_position <= 0,
        on_click=lambda _: set_selected_rows_per_figure(
            row_count_options[row_count_position - 1]
        ),
    )
    next_rows_per_figure = mo.ui.button(
        label="→",
        tooltip="More rows per figure",
        disabled=not row_count_options or row_count_position == len(row_count_options) - 1,
        on_click=lambda _: set_selected_rows_per_figure(
            row_count_options[row_count_position + 1]
        ),
    )
    figure_controls = mo.vstack([
        mo.hstack([previous_time, time_dropdown, next_time]),
        mo.hstack([previous_rows_per_figure, rows_per_figure_dropdown, next_rows_per_figure]),
    ])
    figure_controls
    return current_rows_per_figure, current_time


@app.cell(hide_code=True)
def _(comparison_figures, current_rows_per_figure, current_time, mo, rows):
    figures = (
        comparison_figures(rows, current_time, current_rows_per_figure)
        if current_time and current_rows_per_figure
        else []
    )
    (
        mo.md("No successfully loaded FOVs are available for this time point.")
        if not figures
        else mo.vstack([mo.mpl.interactive(figure) for figure in figures])
    )
    return


@app.cell
def _(mo, rows, status_table, submitted_settings):
    status = (
        mo.md("Submit the form to load FOV images.")
        if submitted_settings is None
        else mo.vstack([mo.md("## FOV status"), status_table(rows)])
    )
    status
    return


if __name__ == "__main__":
    app.run()
