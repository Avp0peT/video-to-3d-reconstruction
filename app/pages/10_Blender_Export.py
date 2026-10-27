"""Generate a verified Blender document from a local, hash-bound PLY."""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys
import uuid

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _path in (_ROOT / "app", _ROOT, _ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def compose_spec(ply: pathlib.Path, *, example: bool = False) -> dict:
    """Bind the actual PLY bytes; geometry validation belongs to the exporter."""
    ply = ply.expanduser().resolve(strict=True)
    if not ply.is_file() or ply.suffix.lower() != ".ply":
        raise ValueError("请选择存在的 PLY 文件。")
    # Both profiles have the same file budget. Read at most one byte beyond it.
    with ply.open("rb") as stream:
        content = stream.read(16 * 1024 * 1024 + 1)
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("PLY 超过 16 MiB 文件上限。")
    note = "本地 PLY 导出；保留输入面与孔洞，尺度未知。导出检查不证明照片保真或物理精度。"
    if example:
        note += ("当前样例源为 component0/QEM 简化重建；4 个删面缺口来自 "
                 "repair_created_excision，非物体真实孔洞；不是闭合实体。")
    return {
        "schema": "ply-viewer-scenes-v1",
        "scenes": [{
            "ply": str(ply), "sha256": hashlib.sha256(content).hexdigest(),
            "title": "开放修复样例" if example else ply.stem[:160],
            "data_kind": "real_photos" if example else "unspecified",
            "declared_unit": None, "note": note,
        }],
    }


def verified_download(result: dict, output: pathlib.Path) -> pathlib.Path | None:
    """Only expose the successful backend artifact with its bound bytes intact."""
    if result.get("status") != "SUCCESS":
        return None
    stages = result.get("stages", [])
    if not stages or stages[-1].get("data", {}).get("status") != "PASS":
        return None
    artifact = result.get("blend", {})
    path = pathlib.Path(artifact.get("path", "")).resolve()
    if path != (output / "model.blend").resolve() or not path.is_file():
        return None
    if path.stat().st_size != artifact.get("bytes"):
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get("sha256"):
        return None
    return path


def render() -> None:
    import streamlit as st
    import _shared as ui

    ui.setup_page("生成 Blender 文件")
    ui.title("生成 Blender 文件", "选择本地 PLY，系统生成可直接打开的 .blend。")
    st.caption("保留输入孔洞，尺度未知。三角网格和点云都可导出。")
    sample = (_ROOT / "outputs/delegation/PROJECT_FINISH/20261011_open_repair_A"
              "/candidate01/open_surface.ply")
    if st.button("使用当前已验收样例", disabled=not sample.is_file()):
        st.session_state["blender_ply_path"] = str(sample)
    ply_text = st.text_input("本地 PLY 路径", key="blender_ply_path", value="")
    blender_text = st.text_input("Blender 程序路径（可选）", value="",
                                placeholder="留空由系统查找已安装的 Blender")
    profile = st.selectbox("导出范围", ["standard", "surface400k"],
                          format_func=lambda p: {
                              "standard": "标准：最多 10 万点 / 20 万面",
                              "surface400k": "大表面：最多 20 万点 / 40 万面",
                          }[p])
    st.caption("两种范围均限制 PLY 为 16 MiB；不会自动提高上限。")
    if st.button("生成 Blender 文件", type="primary", disabled=not ply_text.strip()):
        st.session_state.pop("blender_export_result", None)
        task = None
        try:
            # Import only when requested; missing backend remains a visible error.
            from integrations.blender_export.export import export_blend

            ply = pathlib.Path(ply_text.strip()).expanduser().resolve(strict=True)
            spec = compose_spec(ply, example=ply == sample.resolve())
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            task = _ROOT / "outputs/blender" / f"{stamp}_{uuid.uuid4().hex[:12]}"
            task.mkdir(parents=True, exist_ok=False)
            spec_path = task / "SCENES.json"
            spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
            output = task / "export"
            with st.spinner("正在生成并回读检查 Blender 文件…"):
                result = export_blend(spec_path, output,
                                      blender_exe=pathlib.Path(blender_text.strip())
                                      if blender_text.strip() else None,
                                      profile=profile)
            st.session_state["blender_export_result"] = {
                "result": result, "output": str(output), "task": str(task),
            }
        except ImportError as error:
            st.error(f"Blender 导出后端或依赖暂不可用：{error}。请检查本机环境，不会自动安装。")
        except Exception as error:
            st.error(f"生成未完成：{error}")
            if task is not None:
                error_record = {"status": "ERROR", "error": str(error)}
                (task / "UI_ERROR.json").write_text(
                    json.dumps(error_record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
                st.caption(f"记录保留在：{task}")
    saved = st.session_state.get("blender_export_result")
    if saved:
        result = saved["result"]
        try:
            blend = verified_download(result, pathlib.Path(saved["output"]))
        except (OSError, TypeError, ValueError):
            blend = None
        if blend is not None:
            st.success("Blender 文件已生成，独立回读检查通过。")
            scenes = result.get("scenes", [])
            for scene in scenes:
                count = scene.get("faces", 0)
                st.write(f"{scene.get('vertices', 0):,} 点 · {count:,} 面 · "
                         f"{'三角网格' if count else '点云'} · 尺度未知")
            st.caption("保留输入孔洞。导出不会补洞或生成闭合实体。")
            st.download_button("下载 Blender 文件", blend.read_bytes(),
                               file_name="model.blend", mime="application/octet-stream")
        else:
            st.error("未取得通过回读检查的 Blender 文件。" + str(result.get("error", "")))
        st.caption(f"文件与记录保留在：{saved['task']}")
        with st.expander("查看生成记录"):
            st.json(result)
    ui.footer()


if __name__ == "__main__":
    render()
