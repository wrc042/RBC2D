import numpy as np
import os
import shutil
import sys
import time
import matplotlib.pyplot as plt

# - exp_name
#     - frame.txt
#     - log.txt
#     - time.txt
#     - config.toml TODO
#     - mesh
#         - {name}.ply
#         - {name}.dat TODO
#         - {name}
#             - {name}-{seq:04d}.ply
#             - {name}-{seq:04d}.dat TODO
#     - particle TODO
#         - {name}-{seq:04d}.dat
#     - img
#         - {name}.png
#         - {name}
#             - {name}-{seq:04d}.png
#     - src


class Exporter:
    _inited = False
    _root_dir = ""
    _exp_name = ""
    _exp_dir = ""
    _frame_path = ""
    _frame_tag = -1
    _log_path = ""
    _time_path = ""
    _mesh_dir = ""
    _img_dir = ""
    _src_dir = ""
    _t_st = 0
    _t_ft = None
    _timer_st = {}
    _timer_acc = {}
    _timer_cnt = {}
    _timer_cnted = {}

    @staticmethod
    def init(root_dir="exps", exp_name="output", time_suffix=False):
        if Exporter._inited:
            print("[EXPIO] Exporter already initialized.")
            return
        Exporter._inited = True
        Exporter._root_dir = root_dir
        Exporter._exp_name = exp_name
        Exporter._exp_dir = os.path.join(Exporter._root_dir, Exporter._exp_name)
        log_str = f"[EXPIO] Export to {Exporter._exp_dir}"
        if time_suffix:
            Exporter._exp_dir += "-" + time.strftime("%m%d-%H%M%S")
        if os.path.exists(Exporter._exp_dir):
            with open(os.path.join(Exporter._exp_dir, "time.txt"), "r") as f:
                timestr = f.read()
            redir = os.path.join(Exporter._root_dir, timestr + "-" + Exporter._exp_name)
            shutil.move(Exporter._exp_dir, redir)
            log_str += f"\n[EXPIO] Directory exists; moved to {redir}"
        os.makedirs(Exporter._exp_dir)

        Exporter._frame_path = os.path.join(Exporter._exp_dir, "frame.txt")
        Exporter._frame_tag = -1
        with open(Exporter._frame_path, "w") as f:
            f.write(str(Exporter._frame_tag))

        Exporter._log_path = os.path.join(Exporter._exp_dir, "log.txt")
        Exporter.log(log_str)

        Exporter._time_path = os.path.join(Exporter._exp_dir, "time.txt")
        with open(Exporter._time_path, "w") as f:
            f.write(time.strftime("%m%d-%H%M%S"))

        Exporter._mesh_dir = os.path.join(Exporter._exp_dir, "mesh")

        Exporter._src_dir = os.path.join(Exporter._exp_dir, "src")

        Exporter._img_dir = os.path.join(Exporter._exp_dir, "img")

        Exporter._t_st = time.time()
        Exporter._t_ft = Exporter._t_st

    @staticmethod
    def tag_frame(frame=-1, log_time=True):
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        if not frame == -1:
            Exporter._frame_tag = frame
        else:
            Exporter._frame_tag += 1
        with open(Exporter._frame_path, "w") as f:
            f.write(str(Exporter._frame_tag))
        txt = f"[EXPIO] Tag frame {Exporter._frame_tag}"
        if log_time:
            t_ed = time.time()
            tf = t_ed - Exporter._t_ft
            Exporter._t_ft = t_ed
            ta = t_ed - Exporter._t_st
            txt += f" ({tf:.4f}s / {ta:.4f}s)"
        txt += "\n"
        Exporter.log(txt)

    @staticmethod
    def log(msg):
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        with open(Exporter._log_path, "a") as f:
            f.write(msg + "\n")
        print(f"{msg}")

    @staticmethod
    def export_mesh(mesh, name, frame=None, suffix="ply"):
        t0 = time.time()
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return

        if suffix not in ["ply", "obj"]:
            print(f"[EXPIO] Unknown mesh format {suffix}")
            return
        if frame is None:
            subdir = Exporter._mesh_dir
            path = os.path.join(subdir, f"{name}.{suffix}")
        else:
            if frame == -1:
                frame = Exporter._frame_tag + 1
            subdir = os.path.join(Exporter._mesh_dir, name)
            path = os.path.join(subdir, f"{name}-{frame:04d}.{suffix}")
        os.makedirs(subdir, exist_ok=True)
        mesh.export(path)
        t1 = time.time()
        ta = (t1 - t0) * 1000
        Exporter.log(f"[EXPIO] Export mesh: {path} ({ta:.2f}ms)")

    @staticmethod
    def export_field_image(
        field, name, frame=None, relative=True, vmin=0.0, vmax=1.0, cmap="jet", dpi=100
    ):
        t0 = time.time()
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        if frame is None:
            subdir = Exporter._img_dir
            path = os.path.join(subdir, f"{name}.png")
        else:
            if frame == -1:
                frame = Exporter._frame_tag + 1
            subdir = os.path.join(Exporter._img_dir, name)
            path = os.path.join(subdir, f"{name}-{frame:04d}.png")
        os.makedirs(subdir, exist_ok=True)

        array = field.copy().transpose()
        figx = 10
        figy = array.shape[0] / array.shape[1] * figx
        fig = plt.figure(figsize=(figx, figy), clear=True)
        ax = fig.add_subplot()
        ax.set_axis_off()
        ax.set_xlim([0, array.shape[1]])
        ax.set_ylim([0, array.shape[0]])
        if relative:
            p = ax.imshow(array, cmap=cmap)
        else:
            p = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)

        fig.colorbar(p)
        if frame is not None:
            plt.title(f"frame {frame:04d}")
        fig.savefig(path, dpi=dpi)
        plt.close()
        t1 = time.time()
        ta = (t1 - t0) * 1000
        Exporter.log(f"[EXPIO] Export image: {path} ({ta:.2f}ms)")

    @staticmethod
    def export_contour_image(
        phi, name, frame=None, color=["#03A9F4", "#1976D2"], dpi=100
    ):
        t0 = time.time()
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        if frame is None:
            subdir = Exporter._img_dir
            path = os.path.join(subdir, f"{name}.png")
        else:
            if frame == -1:
                frame = Exporter._frame_tag + 1
            subdir = os.path.join(Exporter._img_dir, name)
            path = os.path.join(subdir, f"{name}-{frame:04d}.png")
        os.makedirs(subdir, exist_ok=True)

        array = phi.copy().transpose()
        figx = 10
        figy = array.shape[0] / array.shape[1] * figx
        fig = plt.figure(figsize=(figx, figy), clear=True)
        ax = fig.add_subplot()

        ax.contourf(array, levels=[-1e20, 0.0], colors=color[0])
        ax.contour(array, levels=[0], colors=color[1], linewidths=1)

        plt.tick_params(axis="both", which="both", length=0)
        plt.xticks([])
        plt.yticks([])

        if frame is not None:
            plt.title(f"frame {frame:04d}")
        fig.savefig(path, dpi=dpi)
        plt.close()
        t1 = time.time()
        ta = (t1 - t0) * 1000
        Exporter.log(f"[EXPIO] Export image: {path} ({ta:.2f}ms)")

    @staticmethod
    def export_ptcl_image(
        ptcl, name, frame=None, domain=None, s=1, seq_color=False, dpi=100
    ):
        t0 = time.time()
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        if frame is None:
            subdir = Exporter._img_dir
            path = os.path.join(subdir, f"{name}.png")
        else:
            if frame == -1:
                frame = Exporter._frame_tag + 1
            subdir = os.path.join(Exporter._img_dir, name)
            path = os.path.join(subdir, f"{name}-{frame:04d}.png")
        os.makedirs(subdir, exist_ok=True)

        if domain is not None:
            figx = 10
            xrange = domain[0][1] - domain[0][0]
            yrange = domain[1][1] - domain[1][0]
            figy = yrange / xrange * figx
            fig = plt.figure(figsize=(figx, figy), clear=True)
            ax = fig.add_subplot()

            ax.set_xlim(domain[0])
            ax.set_ylim(domain[1])
        else:
            fig = plt.figure(figsize=(10, 10), clear=True)
            ax = fig.add_subplot()

            xrange = np.max(ptcl[:, 0]) - np.min(ptcl[:, 0])
            yrange = np.max(ptcl[:, 1]) - np.min(ptcl[:, 1])
            ax.set_xlim(
                [np.min(ptcl[:, 0]) - xrange * 0.1, np.max(ptcl[:, 0]) + xrange * 0.1]
            )
            ax.set_ylim(
                [np.min(ptcl[:, 1]) - yrange * 0.1, np.max(ptcl[:, 1]) + yrange * 0.1]
            )

        if seq_color:
            color = plt.cm.coolwarm(np.linspace(0, 1, len(ptcl)))
        else:
            color = "#03A9F4"

        ax.scatter(ptcl[:, 0], ptcl[:, 1], c=color, s=s)

        if frame is not None:
            plt.title(f"frame {frame:04d}")
        fig.savefig(path, dpi=dpi)
        plt.close()
        t1 = time.time()
        ta = (t1 - t0) * 1000
        Exporter.log(f"[EXPIO] Export image: {path} ({ta:.2f}ms)")

    @staticmethod
    def get_image_path(
        name,
        frame=None,
    ):
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        if frame is None:
            subdir = Exporter._img_dir
            path = os.path.join(subdir, f"{name}.png")
        else:
            if frame == -1:
                frame = Exporter._frame_tag + 1
            subdir = os.path.join(Exporter._img_dir, name)
            path = os.path.join(subdir, f"{name}-{frame:04d}.png")
        os.makedirs(subdir, exist_ok=True)
        Exporter.log(f"[EXPIO] Export image: {path}")
        return path

    @staticmethod
    def copy_src(file):
        if not Exporter._inited:
            print("[EXPIO] Exporter not initialized.")
            return
        if not os.path.exists(file):
            print(f"[EXPIO] File {file} not found.")
            return
        shutil.copy(file, Exporter._src_dir)

    # Timer
    @staticmethod
    def tick(name):
        if name in Exporter._timer_st:
            print(f'[EXPIO] Timer "{name}" already started.')
            return
        Exporter._timer_st[name] = time.perf_counter()
        Exporter._timer_cnt[name] = Exporter._timer_cnt.get(name, 0) + 1

    @staticmethod
    def tock(name, skip_cnt=0, verbose=True):
        if name not in Exporter._timer_st:
            print(f'[EXPIO] Timer "{name}" not started.')
            return
        dt = time.perf_counter() - Exporter._timer_st.pop(name)
        if skip_cnt >= Exporter._timer_cnt[name]:
            if verbose:
                Exporter.log(
                    f'[EXPIO][Timer] "{name}" skipped ({Exporter._timer_cnt[name]} / {skip_cnt})'
                )
            return
        Exporter._timer_acc[name] = Exporter._timer_acc.get(name, 0) + dt
        Exporter._timer_cnted[name] = Exporter._timer_cnted.get(name, 0) + 1
        avgt = Exporter._timer_acc[name] / Exporter._timer_cnted[name]
        if verbose:
            Exporter.log(
                f'[EXPIO][Timer] "{name}": ({dt:.3e}s / {Exporter._timer_acc[name]:.3e}s [{avgt:.3e}s x {Exporter._timer_cnted[name]}])'
            )
