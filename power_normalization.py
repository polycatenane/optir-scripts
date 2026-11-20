# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.24.0", "matplotlib>=3.11.1", "numpy>=2.0", "pandas>=2.3"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # IR laser intensity

    This notebook reads photodiode measurements of the IR laser and averages each scan
    to give one intensity value per wavenumber.

    It then uses a Fourier transform to remove variations shorter than the selected
    wavenumber period. The filtered values are saved.

    `notebook.py` expects an older power file format, so the saved CSV stores these
    intensity values in the `Averaged_Power_mV` column.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    from hyperspectral_cells.power import (
        low_pass_fft,
        power_output_path,
        power_table,
        read_ac_power,
        write_power,
    )

    DATA_ROOT = Path(
        "~/Documents/Bai/Data/2026-11-11 - Starches Redo-20261113T102145Z-1-001/2026-11-11 - Starches Redo"
    ).expanduser()
    INPUT_DIR = DATA_ROOT / "Power"
    OUTPUT_DIR = DATA_ROOT / "Power data Processed"
    DEFAULT_SHORTEST_PERIOD_CM = 50
    return (
        DEFAULT_SHORTEST_PERIOD_CM,
        INPUT_DIR,
        OUTPUT_DIR,
        low_pass_fft,
        mo,
        plt,
        power_output_path,
        power_table,
        read_ac_power,
        write_power,
    )


@app.cell
def _(INPUT_DIR, OUTPUT_DIR, mo, power_output_path, read_ac_power):
    diagnostics = read_ac_power(INPUT_DIR)
    output_path = power_output_path(OUTPUT_DIR, diagnostics.attrs["date"])
    x = diagnostics["Wavenumber_cm-1"]
    mo.md(f"""
    **Input:** `{INPUT_DIR}`  
    **Date:** `{diagnostics.attrs["date"]}`  
    **Scans:** {len(diagnostics)}, from {x.min():g} to {x.max():g} cm⁻¹  
    **Output:** `{output_path}`
    """)
    return (diagnostics,)


@app.cell
def _(diagnostics, mo, plt):
    _figure, _axis = plt.subplots(figsize=(10, 4))
    _x = diagnostics["Wavenumber_cm-1"]
    _mean = diagnostics["Averaged_Power_mV"]
    _std = diagnostics["std"]
    _axis.plot(_x, _mean, color="tab:blue", label="Mean intensity")
    _axis.fill_between(_x, _mean - _std, _mean + _std, color="tab:blue", alpha=0.25, label="±1 SD")
    _axis.set_xlabel("Wavenumber (cm⁻¹)")
    _axis.set_ylabel("Photodiode signal (mV)")
    _axis.legend()
    _axis.grid(alpha=0.25)
    mo.mpl.interactive(_figure)
    return


@app.cell
def _(diagnostics, mo):
    mo.ui.table(diagnostics, selection=None, label="Scan summary")
    return


@app.cell
def _(DEFAULT_SHORTEST_PERIOD_CM, mo):
    period_options = [str(period) for period in range(10, 205, 5)]
    selected_period, set_selected_period = mo.state(
        str(DEFAULT_SHORTEST_PERIOD_CM), allow_self_loops=True
    )
    return period_options, selected_period, set_selected_period


@app.cell
def _(mo, period_options, selected_period, set_selected_period):
    current_period = selected_period()
    period_position = period_options.index(current_period)
    period_dropdown = mo.ui.dropdown(
        options=period_options,
        value=current_period,
        label="Shortest variation to keep (cm-1)",
        on_change=set_selected_period,
    )
    previous_period = mo.ui.button(
        label="←",
        tooltip="Shorter variation",
        disabled=period_position == 0,
        on_click=lambda _: set_selected_period(period_options[period_position - 1]),
    )
    next_period = mo.ui.button(
        label="→",
        tooltip="Longer variation",
        disabled=period_position == len(period_options) - 1,
        on_click=lambda _: set_selected_period(period_options[period_position + 1]),
    )
    mo.hstack([previous_period, period_dropdown, next_period])
    return (current_period,)


@app.cell
def _(current_period, diagnostics, low_pass_fft):
    filtered = low_pass_fft(diagnostics, float(current_period))
    return (filtered,)


@app.cell
def _(filtered, plt):
    _figure, _axis = plt.subplots(figsize=(10, 4))
    _x = filtered["Wavenumber_cm-1"]
    _axis.plot(_x, filtered["raw_mV"], color="tab:gray", alpha=0.6, label="Mean intensity")
    _axis.plot(_x, filtered["Averaged_Power_mV"], color="tab:red", label="Filtered intensity")
    _axis.set_xlabel("Wavenumber (cm⁻¹)")
    _axis.set_ylabel("Photodiode signal (mV)")
    _axis.legend()
    _axis.grid(alpha=0.25)
    plt.close(_figure)
    _figure
    return


@app.cell
def _(filtered, mo, power_table):
    mo.ui.table(power_table(filtered), selection=None, label="File preview")
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save intensity file")
    save_button
    return (save_button,)


@app.cell
def _(OUTPUT_DIR, filtered, mo, save_button, write_power):
    mo.stop(not save_button.value)
    saved_path = write_power(filtered, OUTPUT_DIR)
    mo.md(f"Saved `{saved_path}`")
    return


if __name__ == "__main__":
    app.run()
