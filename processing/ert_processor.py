import os
import numpy as np
import matplotlib.pyplot as plt

# Try to import PyGIMLi; if unavailable, fallback to synthetic processing
try:
    import pygimli as pg
    import pygimli.physics.ert as ert
    HAS_PYGIMLI = True
except Exception:
    HAS_PYGIMLI = False

def process_ert_data(file_path, output_dir='/tmp'):
    """
    If PyGIMLi is available:
        - Load .dat/.xyz, invert, and export result image + CSV.
    Else:
        - Fallback: parse simple XYZ or CSV grid and render a heatmap,
          or generate a synthetic inversion-like image for demo.
    Returns: (image_path, model_csv_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    img_out = os.path.join(output_dir, 'ert_result.png')
    csv_out = os.path.join(output_dir, 'ert_model.csv')

    if HAS_PYGIMLI:
        try:
            data = ert.load(file_path)
            mgr = ert.ERTManager(data)
            mgr.invert(verbose=False)
            # Save image
            fig, ax = plt.subplots(figsize=(8,4))
            mgr.showResult(ax=ax, logScale=True)
            ax.set_title('ERT Inversion (PyGIMLi)')
            fig.savefig(img_out, bbox_inches='tight')
            plt.close(fig)
            # Save model vector
            try:
                # Save as numpy txt for portability
                np.savetxt(csv_out, np.asarray(mgr.model), delimiter=',')
            except Exception:
                open(csv_out, 'w').write('model_length,' + str(len(mgr.model)))
            return img_out, csv_out
        except Exception as e:
            # Fall back if PyGIMLi failed to load the file
            print('PyGIMLi processing error:', e)

    # Fallback rendering: read simple numeric grid or synthesize
    try:
        # Attempt to read xyz or csv grid
        arr = None
        if file_path.lower().endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(file_path)
            # Expect columns: x,z,resistivity  (order flexible)
            cols = [c.lower() for c in df.columns]
            x_idx = cols.index('x') if 'x' in cols else 0
            z_idx = cols.index('z') if 'z' in cols else 1
            r_idx = cols.index('resistivity') if 'resistivity' in cols else 2
            x_vals = df.iloc[:, x_idx].to_numpy()
            z_vals = df.iloc[:, z_idx].to_numpy()
            r_vals = df.iloc[:, r_idx].to_numpy()
            # Create grid
            xs = np.unique(x_vals)
            zs = np.unique(z_vals)
            grid = np.full((len(zs), len(xs)), np.nan)
            for x, z, r in zip(x_vals, z_vals, r_vals):
                xi = np.where(xs==x)[0][0]
                zi = np.where(zs==z)[0][0]
                grid[zi, xi] = r
            arr = grid
        else:
            # Generate synthetic resistivity
            nx, nz = 80, 40
            x = np.linspace(0, 1, nx)
            z = np.linspace(0, 1, nz)
            X, Z = np.meshgrid(x, z)
            arr = 30 + 70*np.exp(-((X-0.5)**2)/(0.02) - ((Z-0.6)**2)/(0.03)) + 10*np.sin(8*X)*np.exp(-3*Z)
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8,4))
        im = ax.imshow(arr, origin='lower', aspect='auto')
        ax.set_title('ERT Pseudo-Section (Fallback Renderer)')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Depth')
        fig.colorbar(im, ax=ax, label='Resistivity (Ohm·m)')
        fig.savefig(img_out, bbox_inches='tight')
        plt.close(fig)
        # Save numeric array
        np.savetxt(csv_out, np.asarray(arr), delimiter=',')
        return img_out, csv_out
    except Exception as e:
        print('Fallback ERT render error:', e)
        return None, None
