"""Bounded photos/video pipeline UI; native work stays in the supervised CLI."""
from __future__ import annotations
import datetime
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT, ROOT / "src", ROOT / "app"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
LIGHT_PYTHON = ROOT / "outputs/delegation/T02A/envs/20261002T190239Z_9b9b5ddb/venv/Scripts/python.exe"


def verified_blend(state: dict, run: Path) -> Path | None:
    if (state.get("deliverable_ready") is not True and not
            (state.get("target") == "cad" and state.get("geometry_deliverable_ready") is True)):
        return None
    blender = state.get("stages", {}).get("blender", {})
    if blender.get("status") != "SUCCESS" or blender.get("reopen_verified") is not True:
        return None
    ref = blender.get("blend", blender)
    path = Path(ref.get("path", "")).resolve()
    if not path.is_relative_to(run.resolve()) or not path.is_file() or path.suffix.lower() != ".blend":
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return path if digest.hexdigest() == ref.get("sha256") else None


def task_directory(value: str) -> Path:
    """Resolve before reading; junction/symlink escapes are rejected."""
    run = Path(value).expanduser().resolve(strict=True)
    outputs = (ROOT / "outputs").resolve(strict=True)
    if not run.is_dir() or not run.is_relative_to(outputs) or run == outputs:
        raise ValueError("请选择本项目 outputs 内的任务 pipeline 目录。")
    state_path = (run / ("pipeline.json" if (run / "pipeline.json").is_file() else "CONTINUATION.json")).resolve()
    if not state_path.is_relative_to(run) or not state_path.is_file():
        raise ValueError("目录中没有本任务的 pipeline.json 或 CONTINUATION.json。")
    return run


def load_existing(value: str) -> dict:
    run = task_directory(value)
    if (run / "CONTINUATION.json").is_file() and not (run / "pipeline.json").is_file():
        return {"task": str(run.parent), "run": str(run), "supervision": {},
                "loaded_existing": True, "continuation": True}
    supervision = {}
    wrapper = (run.parent / "UI_SUPERVISION.json").resolve()
    if run.name == "pipeline" and wrapper.is_relative_to(run.parent) and wrapper.is_file():
        candidate = json.loads(wrapper.read_text(encoding="utf-8"))
        if not isinstance(candidate, dict):
            raise ValueError("任务运行记录格式不完整。")
        argv = candidate.get("argv", [])
        if isinstance(argv, list) and "--output" in argv:
            index = argv.index("--output")
            if index + 1 < len(argv) and Path(argv[index + 1]).resolve() == run:
                supervision = candidate
    return {"task": str(run.parent), "run": str(run), "supervision": supervision,
            "loaded_existing": True, "stage_evidence_verified": not supervision and verify_stage_evidence(run)}


def verify_stage_evidence(run: Path) -> bool:
    """CLI records prove individual stages, never an absent outer supervisor."""
    try:
        run = task_directory(str(run))
        state = json.loads((run / "pipeline.json").read_text(encoding="utf-8"))
        geometry_only = (state.get("target") == "cad" and state.get("geometry_deliverable_ready") is True
                         and state.get("status") in ("SUCCESS", "PARTIAL", "FAILED"))
        if not geometry_only and (state.get("status") not in ("SUCCESS", "PARTIAL") or state.get("exit_code") not in (0, 1)
                or state.get("deliverable_ready") is not True or state.get("errors") != []):
            return False
        names = ("input", "sfm", "segmentation", "native_subset", "mvs_bridge", "mvs_masks",
                 "surface_plan", "surface_reconstruction", "blender_export")
        for name in names:
            stage = state["stages"][name]
            allowed = (0, 1) if name in ("sfm", "segmentation") else (0,)
            if stage.get("exit_code") not in allowed:
                return False
            expected = (run / "logs" / name / "ENTRY.json").resolve()
            if not expected.is_relative_to(run) or Path(stage["entry_path"]).resolve() != expected or not expected.is_file():
                return False
            entry = json.loads(expected.read_text(encoding="utf-8"))
            if (entry.get("status") != "completed" or entry.get("reclaim_confirmed") is not True
                    or entry.get("errors") != [] or entry.get("log_errors") != []
                    or not isinstance(stage.get("argv"), list) or not stage["argv"]
                    or entry.get("argv") != stage["argv"]
                    or entry.get("root_exit_code") != stage["exit_code"]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def execute(source: str, prompt: str, frames: int, model_id: str, target: str = "blender") -> dict:
    from integrations.mvs_surface.supervision import run_supervised
    if target not in ("blender", "cad"):
        raise ValueError("不支持的交付目标。")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task = ROOT / "outputs/photo_ui" / f"{stamp}_{uuid.uuid4().hex[:12]}"
    task.mkdir(parents=True, exist_ok=False)
    evidence = task / "cli"
    evidence.mkdir()
    run = task / "pipeline"
    argv = [str(LIGHT_PYTHON), "-B", "-X", "utf8", str(ROOT / "scripts/run_photo_demo.py"),
            "run", "--source", source, "--prompt", prompt, "--frames", str(frames),
            "--target", target, "--output", str(run), "--python", str(LIGHT_PYTHON)]
    if model_id.strip():
        argv.extend(["--model-id", model_id.strip()])
    try:
        result = run_supervised(argv, cwd=str(task), output=str(evidence), wall_seconds=1800,
                                memory_bytes=12 * 1024**3, output_bytes=3 * 1024**3,
                                label="photo_ui_pipeline")
    except Exception as exc:
        result = {"status": "ERROR", "errors": [f"{type(exc).__name__}: {exc}"],
                  "reclaim_confirmed": False}
    (task / "UI_SUPERVISION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"task": str(task), "run": str(run), "supervision": result}


def execute_cad(run: Path) -> dict:
    """Explicit continuation only; never reruns photos, SAM3, or reconstruction."""
    from integrations.demo_pipeline.cad import plan as cad_plan
    from integrations.mvs_surface.supervision import run_supervised
    run = task_directory(str(run))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task = ROOT / "outputs/photo_cad_ui" / f"{stamp}_{uuid.uuid4().hex[:12]}"
    output = task / "delivery"
    cad_plan(run, output, python=LIGHT_PYTHON)  # Read-only, before native launch.
    evidence = task / "cli"
    evidence.mkdir(parents=True, exist_ok=False)
    argv = [str(LIGHT_PYTHON), "-B", "-X", "utf8", str(ROOT / "scripts/continue_photo_cad.py"),
            "--run", str(run), "--output", str(output), "--python", str(LIGHT_PYTHON)]
    try:
        result = run_supervised(argv, cwd=str(task), output=str(evidence), wall_seconds=240,
                                memory_bytes=3 * 1024**3, output_bytes=512 * 1024**2,
                                label="photo_ui_cad")
    except Exception as exc:
        result = {"status": "ERROR", "errors": [f"{type(exc).__name__}: {exc}"],
                  "reclaim_confirmed": False}
    (task / "UI_SUPERVISION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_run": str(run), "task": str(task), "output": str(output), "supervision": result}


def verified_cad(saved: dict, run: Path) -> dict | None:
    from integrations.demo_pipeline.cad import verify_cad_output
    supervisor = saved.get("supervision", {})
    if (Path(saved.get("source_run", "")).resolve() != run.resolve()
            or supervisor.get("status") != "completed" or type(supervisor.get("root_exit_code")) is not int
            or supervisor["root_exit_code"] != 0 or supervisor.get("reclaim_confirmed") is not True
            or supervisor.get("errors") or supervisor.get("log_errors")):
        return None
    return verify_cad_output(Path(saved["output"]), run)


def cad_download_payload(reference: dict) -> bytes:
    """Check the actual bytes handed to Streamlit, not a previous file read."""
    count = reference.get("bytes")
    if type(count) is not int or not 0 < count <= 256 * 1024**2:
        raise ValueError("CAD 文件大小声明无效。")
    with Path(reference["path"]).open("rb") as stream:
        payload = stream.read(count + 1)
    if len(payload) != count or hashlib.sha256(payload).hexdigest() != reference.get("sha256"):
        raise ValueError("CAD 文件在核验后发生变化，拒绝下载。")
    return payload


def unified_cad(state: dict, run: Path) -> dict | None:
    from integrations.demo_pipeline.cad import verify_cad_output
    stage = state.get('stages', {}).get('cad', {})
    if state.get('target') != 'cad' or stage.get('status') != 'SUCCESS':
        return None
    output = (run / 'cad').resolve()
    if Path(stage.get('output', '')).resolve() != output:
        raise ValueError('CAD 输出未绑定本任务。')
    return verify_cad_output(output, run)


def surface_resume_plan(run: Path, output: Path) -> dict:
    """Read-only eligibility; never backfill historical checkpoint evidence."""
    from integrations.demo_pipeline.resume import plan as resume_plan
    from integrations.demo_pipeline.surface import read
    run = task_directory(str(run))
    if not (run / 'SURFACE_RESUME_SOURCE.json').is_file():
        raise ValueError('来源没有生产时正式续跑 checkpoint；历史任务不能补造。')
    state = read(run / 'pipeline.json')
    if state.get('status') == 'RUNNING' or state.get('exit_code') is None:
        raise ValueError('来源尚未结束，不能续跑。')
    old_plan = read(run / 'PLAN.json')
    return resume_plan(run, output, python=LIGHT_PYTHON,
                       tool_root=old_plan['mvs_tool_root'], blender_exe=old_plan['blender_exe'],
                       model_id=old_plan.get('model_id'), root=ROOT)


def execute_surface_resume(run: Path) -> dict:
    """Explicit click only; launch the existing resume CLI in a fresh task."""
    from integrations.mvs_surface.supervision import run_supervised
    run = task_directory(str(run))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    task = ROOT / 'outputs/photo_resume_ui' / f'{stamp}_{uuid.uuid4().hex[:12]}'
    output = task / 'continuation'
    frozen = surface_resume_plan(run, output)
    evidence = task / 'cli'; evidence.mkdir(parents=True, exist_ok=False)
    argv = [str(LIGHT_PYTHON), '-B', '-X', 'utf8', str(ROOT / 'scripts/run_photo_demo.py'),
            'resume', '--run', str(run), '--output', str(output), '--python', frozen['python'],
            '--mvs-tool-root', frozen['tool_root'], '--blender-exe', frozen['blender_exe']]
    if frozen.get('requested_model_id') is not None:
        argv += ['--model-id', frozen['requested_model_id']]
    try:
        outcome = run_supervised(argv, cwd=str(task), output=str(evidence), wall_seconds=1800,
                                 memory_bytes=12 << 30, output_bytes=3 << 30, label='photo_ui_surface_resume')
    except Exception as exc:
        outcome = {'status': 'ERROR', 'errors': [f'{type(exc).__name__}: {exc}'], 'reclaim_confirmed': False}
    (task / 'UI_SUPERVISION.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'task': str(task), 'run': str(output), 'source_run': str(run),
            'supervision': outcome, 'continuation': True}


def continuation_artifacts(run: Path, supervisor: dict | None = None) -> tuple[dict, list[dict]]:
    """Read-only byte/stage/readback verification for continuation downloads."""
    from integrations.demo_pipeline.surface import read
    from integrations.mvs_surface.photo_inputs import _binding
    from integrations.demo_pipeline.pipeline import verify_geometry_outputs
    run = task_directory(str(run))
    state = read(run / 'CONTINUATION.json'); frozen = read(run / 'PLAN.json')
    def require(ok, reason):
        if not ok: raise ValueError(reason)
    require(state.get('schema') == frozen.get('schema') == 'photo-surface-continuation-v1'
            and state.get('status') in ('SUCCESS', 'PARTIAL') and state.get('exit_code') in (0, 1)
            and state.get('deliverable_ready') is True and frozen.get('output') == str(run)
            and state.get('source_run') == frozen.get('source_run')
            and state.get('source_bindings') == frozen.get('bindings'), '续跑尚无完整文件交付或来源绑定不一致。')
    if supervisor:
        require(supervisor.get('status') == 'completed' and supervisor.get('reclaim_confirmed') is True
                and supervisor.get('root_exit_code') == state['exit_code']
                and not supervisor.get('errors') and not supervisor.get('log_errors'), 'UI续跑监督或回收未通过。')
        argv = supervisor.get('argv', [])
        require(isinstance(argv, list) and 'resume' in argv and '--output' in argv
                and Path(argv[argv.index('--output') + 1]).resolve() == run, 'UI监督未绑定该续跑输出。')
    bindings = state['source_bindings']
    require(type(bindings) is list and 0 < len(bindings) <= 2048, '续跑来源文件绑定缺失。')
    for row in bindings:
        require(_binding(row['path'], 512 << 20) == row, '续跑原始来源或依赖文件发生变化。')
    names = ('native_subset', 'mvs_bridge', 'mvs_masks', 'surface_plan', 'surface_reconstruction', 'blender_export')
    require(set(state.get('stages', {})) == set(names), '续跑阶段记录不完整。')
    for name in names:
        detail = state['stages'][name]; expected = run / (name + '_ENTRY.json')
        require(detail.get('status') == 'SUCCESS' and detail.get('exit_code') == 0
                and detail.get('reclaim_confirmed') is True and Path(detail.get('entry_path', '')).resolve() == expected,
                '续跑阶段声明或回收未通过。')
        entry = read(expected)
        require(entry.get('status') == 'completed' and entry.get('root_exit_code') == 0
                and entry.get('reclaim_confirmed') is True and not entry.get('errors') and not entry.get('log_errors')
                and type(entry.get('pid')) is int and entry['pid'] > 0
                and entry.get('cwd') == frozen.get('root') and isinstance(entry.get('argv'), list) and entry['argv'],
                '续跑阶段实际监督未通过。')
    verify_geometry_outputs(state, run / 'geometry')
    surface = state['surface']; blend = state['blender']
    refs = surface.get('artifacts', [])
    require(len(refs) == 1, '续跑表面产物不唯一。')
    evidence = surface.get('evidence_paths', [])
    require(len(evidence) == 1 and Path(evidence[0]).resolve().is_relative_to(run), '表面证据路径无效。')
    native_surface = read(evidence[0])
    require(native_surface.get('status') == 'SURFACE_GENERATED_NOT_ACCEPTED' and native_surface.get('exit_code') == 0,
            '续跑表面生产者未通过。')
    result_path = Path(blend.get('result_path', '')).resolve()
    require(result_path.is_relative_to(run) and result_path.is_file(), 'Blender独立重开记录缺失。')
    exported = read(result_path); stages = exported.get('stages', [])
    require(exported.get('status') == 'SUCCESS' and exported.get('blend') == {k: blend[k] for k in ('path', 'bytes', 'sha256')}
            and exported.get('blend_sha256_before_readback') == blend.get('sha256')
            and [s.get('mode') for s in stages] == ['create', 'verify'], 'Blender文件或独立重开身份不同。')
    require(all(s.get('data', {}).get('status') == 'PASS'
                and s.get('data', {}).get('payload_sha256') == exported.get('payload_sha256')
                and s.get('supervision', {}).get('status') == 'completed'
                and s.get('supervision', {}).get('root_exit_code') == 0
                and s.get('supervision', {}).get('reclaim_confirmed') is True
                and not s.get('supervision', {}).get('errors') and not s.get('supervision', {}).get('log_errors')
                for s in stages), 'Blender创建/重开监督证据未通过。')
    require(stages[0]['data'].get('scenes') == stages[1]['data'].get('scenes')
            and len(stages[1]['data'].get('scenes', [])) == 1
            and stages[1]['data']['scenes'][0].get('source_sha256') == refs[0].get('sha256'), '重开的Blender表面与声明源网格不同。')
    artifacts = refs + [{k: blend[k] for k in ('path', 'bytes', 'sha256')}]
    for ref in artifacts:
        require(_binding(ref['path'], 256 << 20) == ref, '下载文件字节或大小不同。')
    return state, artifacts


def render_continuation(saved: dict):
    import streamlit as st
    st.caption('续跑仅复用已提交输入/SfM/SAM，表面与Blender写入独立新目录。CAD、研究、外形和真实尺度未因此完成。')
    try:
        state, artifacts = continuation_artifacts(Path(saved['run']), saved.get('supervision'))
        st.success('续跑表面和Blender文件已通过字节核验及独立重开；部分视图范围以原记录为准。')
        for ref in artifacts:
            path = Path(ref['path'])
            st.download_button('下载续跑 ' + path.suffix.lstrip('.'), cad_download_payload(ref),
                               file_name=path.name, mime='application/octet-stream', key='resume_' + path.suffix)
        render_continuation_cad(Path(saved['run']))
        with st.expander('续跑来源和阶段状态'): st.json(state)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, IndexError) as exc:
        st.warning(f'续跑文件暂不可下载：{exc}')
        record = Path(saved['run']) / 'CONTINUATION.json'
        if record.is_file():
            from integrations.demo_pipeline.surface import read
            with st.expander('保留的续跑状态'): st.json(read(record))


def render_continuation_cad(run: Path):
    """Explicit CAD continuation of verified surface output; reruns are read-only."""
    import streamlit as st
    from integrations.demo_pipeline.cad import plan as cad_plan
    problem = None
    try:
        cad_plan(run, ROOT/'outputs'/'__cad_continuation_ui_readonly_probe', python=LIGHT_PYTHON)
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        problem = str(exc)
    st.caption('可继续生成分面CAD实体；开放面或孔洞不能通过闭合实体门时，保留原Blender文件。真实尺寸仍待实测。')
    if problem:st.caption('CAD继续条件未满足：' + problem)
    if st.button('从续跑表面生成 CAD 实体',key='continued_surface_cad',
                 disabled=bool(problem) or st.session_state.get('photo3d_cad_busy',False)):
        st.session_state['photo3d_cad_busy']=True
        st.session_state.pop('photo3d_cad_result',None)
        try:
            with st.spinner('正在生成CAD并独立回读…'):
                st.session_state['photo3d_cad_result']=execute_cad(run)
        except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
            st.error('CAD生成未完成：' + str(exc))
        finally:
            st.session_state['photo3d_cad_busy']=False
    saved=st.session_state.get('photo3d_cad_result')
    if not saved or Path(saved.get('source_run','')).resolve()!=run.resolve():return
    try:
        checked=verified_cad(saved,run)
        if checked:
            st.success('CAD实体及独立回读通过；物理尺度和照片精度未验收。')
            for row in checked['artifacts']:
                path=Path(row['path'])
                st.download_button('下载续接 '+path.suffix.lstrip('.'),cad_download_payload(row),
                    file_name=path.name,mime='application/octet-stream',key='continued_cad_'+path.suffix)
        else:
            st.warning('CAD尚未通过全部检查，原表面和Blender保留。')
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        st.warning('CAD下载核验未通过：'+str(exc))
    st.caption('CAD独立任务：'+saved['task'])


def render():
    import streamlit as st
    import _shared as ui
    from integrations.demo_pipeline.pipeline import Options, plan
    ui.setup_page("照片 / 视频生成 3D")
    ui.title("照片 / 视频生成 3D", "选择本地素材，系统依次分割、重建开放表面并生成 .blend。")
    st.caption("开放表面允许孔洞，真实尺度未知。模型可能缺失或粗糙；文件验证通过不代表外形与尺寸准确，请结合原图检查。")
    source = st.text_input("照片目录或视频路径", key="photo3d_source")
    prompt = st.text_input("目标物体描述", value="beverage can", key="photo3d_prompt")
    frames = st.number_input("视频最多抽取帧数", min_value=2, max_value=96, value=24, step=1)
    model_id = st.text_input("SfM 模型 ID（可选）", help="留空选择注册视图最多的分量；不自动拼接分量。")
    with_cad = st.checkbox("同时生成 CAD 实体（FCStd / STEP）", value=False,
                           help="仅闭合且通过检查的表面可转换；失败时保留 Blender，不自动补洞。")
    signature = (source.strip(), prompt.strip(), int(frames), model_id.strip(), 'cad' if with_cad else 'blender')
    ready = bool(signature[0] and signature[1])
    if st.button("检查计划与依赖", disabled=not ready):
        try:
            st.session_state["photo3d_plan"] = {"signature": signature, "value": plan(Options(
                Path(signature[0]), signature[1], frames=signature[2], model_id=signature[3] or None,
                python=LIGHT_PYTHON, target=signature[4]))}
        except Exception as exc:
            st.error(f"计划检查未通过：{exc}")
    saved_plan = st.session_state.get("photo3d_plan")
    if saved_plan and saved_plan["signature"] == signature:
        with st.expander("目标、依赖与执行计划", expanded=True):
            st.json(saved_plan["value"])
    st.caption("每次点击执行创建全新任务；无自动重试。整个 CLI 限时 1800 秒、Job 内存 12 GiB、任务目录 3 GiB；不安装或下载依赖。")
    if st.button("运行整条流程并生成 3D / .blend", type="primary", disabled=not ready or st.session_state.get("photo3d_busy", False)):
        st.session_state["photo3d_busy"] = True
        st.session_state.pop("photo3d_result", None)
        try:
            with st.spinner("正在运行输入处理、重建与分割，并验证 Blender 文件…"):
                st.session_state["photo3d_result"] = execute(*signature)
        except Exception as exc:
            st.error(f"流程未完成：{exc}")
        finally:
            st.session_state["photo3d_busy"] = False
    st.divider()
    st.subheader("载入已有任务")
    existing = st.text_input("已有任务的 pipeline 目录", help="填写本项目 outputs 内含 pipeline.json 或 CONTINUATION.json 的目录；只读取结果，不重新运行。")
    if st.button("载入已有任务", disabled=not existing.strip()):
        try:
            st.session_state["photo3d_result"] = load_existing(existing.strip())
        except (OSError, ValueError, TypeError, KeyError) as exc:
            st.session_state.pop("photo3d_result", None)
            st.error(f"无法载入该任务：{exc}")
    saved = st.session_state.get("photo3d_result")
    if saved:
        run = Path(saved["run"])
        if saved.get('continuation'):
            render_continuation(saved)
            ui.footer()
            return
        st.subheader('继续表面 / Blender')
        eligibility_error = None
        try:
            surface_resume_plan(run, ROOT / 'outputs' / '__surface_resume_ui_readonly_probe')
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            eligibility_error = str(exc)
        if eligibility_error:
            st.caption('续跑不可用：' + eligibility_error)
        else:
            st.caption('生产时checkpoint与上游身份已只读核验；显式点击才在独立新任务继续表面和Blender。')
        if st.button('继续表面 / Blender（复用上游）',
                     disabled=bool(eligibility_error) or st.session_state.get('photo3d_resume_busy', False)):
            st.session_state['photo3d_resume_busy'] = True
            st.session_state.pop('photo3d_resume_result', None)
            try:
                with st.spinner('正在复用已提交上游，生成表面并独立重开Blender…'):
                    st.session_state['photo3d_resume_result'] = execute_surface_resume(run)
            except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
                st.error(f'表面续跑未启动或未完成：{exc}')
            finally:
                st.session_state['photo3d_resume_busy'] = False
        resumed = st.session_state.get('photo3d_resume_result')
        if resumed and Path(resumed.get('source_run', '')).resolve() == run.resolve():
            render_continuation(resumed)
        st.caption(f"任务记录：{saved['task']}；CLI 日志：{Path(saved['task']) / 'cli'}")
        state = {}
        try:
            if saved.get("loaded_existing"):
                run = task_directory(str(run))
            else:
                run = run.resolve()
                if not run.is_relative_to((ROOT / "outputs").resolve()):
                    raise ValueError("任务路径超出本项目 outputs。")
                if not (run / "pipeline.json").resolve().is_relative_to(run):
                    raise ValueError("运行状态路径超出任务目录。")
            if (run / "pipeline.json").is_file():
                state = json.loads((run / "pipeline.json").read_text(encoding="utf-8"))
                if not isinstance(state, dict) or not isinstance(state.get("stages", {}), dict):
                    raise ValueError("任务阶段状态格式不完整。")
            supervisor = saved["supervision"]
            clean = (supervisor.get("status") == "completed"
                     and supervisor.get("root_exit_code") in (0, 1)
                     and state.get("exit_code") == supervisor.get("root_exit_code")
                     and supervisor.get("reclaim_confirmed") is True
                     and not supervisor.get("errors") and not supervisor.get("log_errors"))
            stage_verified = (saved.get("loaded_existing") and not supervisor
                              and verify_stage_evidence(run))
            clean = clean or stage_verified
            if not clean and state.get('target') == 'cad' and state.get('geometry_deliverable_ready') is True:
                # A failed CAD conversion does not invalidate earlier verified geometry.
                clean = verify_stage_evidence(run)
                if clean:
                    from integrations.demo_pipeline.pipeline import verify_geometry_outputs
                    verify_geometry_outputs(state.get('stages', {}), run)
            if clean and saved.get("loaded_existing"):
                from integrations.demo_pipeline.pipeline import verify_geometry_outputs
                stages = state.get("stages", {})
                if stages.get("surface", {}).get("status") != "SUCCESS":
                    clean = False
                else:
                    try:
                        verify_geometry_outputs(stages, run)
                    except (OSError, ValueError, TypeError, RuntimeError):
                        clean = False
            blend = verified_blend(state, run) if clean else None
            from integrations.demo_pipeline.quality import read_for_display, display_message
            object_quality = read_for_display(run, state)
            st.warning(display_message(object_quality))
            if blend:
                if stage_verified:
                    st.caption("各阶段执行记录已核对；此 CLI 任务没有外层整次监督记录。")
                if state.get("status") == "PARTIAL":
                    st.warning("部分视图未覆盖，已生成可查看文件；Blender 重新打开验证通过。")
                else:
                    st.success("文件目标：三维表面与 Blender 文件已生成，重新打开验证通过；物体外形目标另行验收。")
                st.download_button("下载 .blend", blend.read_bytes(), file_name=blend.name, mime="application/octet-stream")
                direct_cad = None
                try:
                    direct_cad = unified_cad(state, run)
                except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                    st.error(f"CAD 记录未通过核验，原 Blender 仍可下载：{exc}")
                if direct_cad:
                    st.success("文件目标：已生成有效闭合 CAD 单实体；闭合不代表全物体外形，真实尺度和实物精度未验证。")
                    for ref in direct_cad['artifacts']:
                        path = Path(ref['path'])
                        st.download_button('下载 ' + path.suffix.lstrip('.'), cad_download_payload(ref),
                                           file_name=path.name, mime='application/octet-stream', key='unified_' + path.suffix)
                    from measurement_ui import render_measurement
                    render_measurement(run, Path(direct_cad['output']))
                elif state.get('target') == 'cad':
                    st.warning("CAD 目标尚未完成，已验证的 Blender 表面保留。")
                st.subheader("继续生成 CAD 实体")
                st.caption("对现有表面做有偏差检查的简化，再验证 FreeCAD 单实体和 STEP 往返。开放或无效网格会停止，不自动补洞。当前没有实测尺度，生成的是任意尺度分面实体。")
                if state.get('target') == 'cad':
                    st.caption("本任务已请求 CAD 阶段，不在原任务中重复执行；失败时请先检查原因，保留现有 Blender 成果。")
                if st.button("生成 FCStd / STEP", disabled=state.get('target') == 'cad' or bool(direct_cad) or st.session_state.get("photo3d_cad_busy", False)):
                    st.session_state["photo3d_cad_busy"] = True
                    st.session_state.pop("photo3d_cad_result", None)
                    try:
                        with st.spinner("正在检查网格、转换实体并独立回读 CAD 文件…"):
                            st.session_state["photo3d_cad_result"] = execute_cad(run)
                    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                        st.error(f"CAD 生成未完成，原三维模型保留：{exc}")
                    finally:
                        st.session_state["photo3d_cad_busy"] = False
                cad_saved = st.session_state.get("photo3d_cad_result")
                if cad_saved and Path(cad_saved["source_run"]).resolve() == run.resolve():
                    try:
                        cad_verified = verified_cad(cad_saved, run)
                        if cad_verified:
                            st.success("有效闭合 CAD 单实体及独立回读已通过；真实尺寸和实物精度未验证。")
                            for row in cad_verified["artifacts"]:
                                path = Path(row["path"])
                                st.download_button("下载 " + path.suffix.lstrip("."), cad_download_payload(row),
                                                   file_name=path.name, mime="application/octet-stream")
                            from measurement_ui import render_measurement
                            render_measurement(run, Path(cad_verified['output']))
                        else:
                            st.warning("CAD 尚未通过全部检查；原 Blender 模型仍可下载。")
                    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
                        st.error(f"CAD 下载核验失败：{exc}")
                    st.caption(f"CAD 独立任务与日志：{cad_saved['task']}")
                    with st.expander("CAD 监督记录"):
                        st.json(cad_saved["supervision"])
            else:
                st.warning("尚无通过全部验证的 .blend。查看阶段状态与保留日志。")
                if saved.get("loaded_existing") and not supervisor and not stage_verified:
                    st.caption("此任务没有可核实的整次运行记录；可查看状态与报告，暂不提供模型下载。")
            report = (run / "report/index.html").resolve()
            if not report.is_relative_to(run):
                raise ValueError("报告路径超出该任务目录。")
            if report.is_file():
                st.download_button("下载离线报告 HTML", report.read_bytes(), file_name="report.html", mime="text/html")
                st.caption(f"完整报告路径：{report}；产物链接需在完整任务目录中打开。")
        except (OSError, ValueError, TypeError) as exc:
            st.error(f"结果核查未通过：{exc}")
        with st.expander("阶段状态和监督记录", expanded=True):
            st.json(state)
            st.json(saved["supervision"])
    ui.footer()


if __name__ == "__main__":
    render()
