"""Small geometry helpers shared by Isaac scene randomization."""


def grid_cell_bounds(workspace_x, workspace_y, shape, cell):
    columns, rows = (int(shape[0]), int(shape[1]))
    column, row = (int(cell[0]), int(cell[1]))
    if columns <= 0 or rows <= 0:
        raise ValueError(f"invalid grid shape: {shape}")
    if not 0 <= column < columns or not 0 <= row < rows:
        raise ValueError(f"cell {(column, row)} is outside grid {(columns, rows)}")
    x_min, x_max = (float(workspace_x[0]), float(workspace_x[1]))
    y_min, y_max = (float(workspace_y[0]), float(workspace_y[1]))
    x_step = (x_max - x_min) / columns
    y_step = (y_max - y_min) / rows
    return (
        (x_min + column * x_step, x_min + (column + 1) * x_step),
        (y_min + row * y_step, y_min + (row + 1) * y_step),
    )
