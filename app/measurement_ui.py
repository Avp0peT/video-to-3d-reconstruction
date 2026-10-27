"""User-provided scale continuation; rendering the panel never starts native work."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import uuid

ROOT = Path(__file__).resolve().parents[1]


def _task(family):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return ROOT / 'outputs' / family / (stamp + '_' + uuid.uuid4().hex[:12])


def preview(run, cad_output):
    from integrations.demo_pipeline import measurement
    source = measurement.inspect_source(run, cad_output)
    return measurement.make_preview(source, _task('photo_measurement_preview'))


def execute_reference(run, cad_output, *, vertex_indices, length, unit, label,
                      correspondence, evidence_path, checks=None):
    from integrations.demo_pipeline import measurement
    from integrations.demo_pipeline.surface import save
    source = measurement.inspect_source(run, cad_output)
    request = measurement.create_request(source, vertex_indices, length, unit, label,
                                         correspondence, evidence_path, checks=checks)
    task = _task('photo_measurement_ui')
    task.mkdir(parents=True, exist_ok=False)
    reference = task / 'reference.json'
    save(reference, request)
    result = {'source_run': str(Path(run).resolve()), 'source_cad_output': str(Path(cad_output).resolve()),
              'task': str(task), 'output': str(task / 'delivery'), 'status': 'PREFLIGHT', 'exit_code': 4}
    try:
        frozen = measurement.plan(run, reference, task / 'delivery', cad_output=cad_output)
        actual = measurement.execute(frozen)
        result.update(status=actual['status'], exit_code=actual['exit_code'])
    except Exception as error:
        result.update(status='FAILED', exit_code=4, error=f'{type(error).__name__}: {error}')
    finally:
        save(task / 'UI_RESULT.json', result)
    return result


def verified_result(saved, run, cad_output):
    from integrations.demo_pipeline import measurement
    if (saved.get('source_run') != str(Path(run).resolve())
            or saved.get('source_cad_output') != str(Path(cad_output).resolve())
            or saved.get('status') != 'SUCCESS' or type(saved.get('exit_code')) is not int
            or saved['exit_code'] != 0):
        return None
    value = measurement.verify_result(saved['output'])
    if (value['source_run'] != str(Path(run).resolve())
            or value['source_cad_output'] != str(Path(cad_output).resolve())):
        raise ValueError('定尺度结果属于另一个源任务。')
    return value


def _payload(row):
    count = row.get('bytes')
    if type(count) is not int or not 0 < count <= 256 << 20:
        raise ValueError('下载大小声明无效。')
    with Path(row['path']).open('rb') as stream:
        payload = stream.read(count + 1)
    if len(payload) != count or hashlib.sha256(payload).hexdigest() != row['sha256']:
        raise ValueError('下载文件已经改变。')
    return payload


def independent_check(first, second, length, unit, label, correspondence, proof, tolerance):
    """Bind the user's check record; the backend validates geometry and numbers."""
    from integrations.mesh_scale.ledger import file_binding
    if not all(value.strip() for value in (length, label, correspondence, proof, tolerance)):
        raise ValueError('请填写独立校验长度、名称、端点对应说明和允许误差。')
    path = Path(proof).resolve(strict=True)
    if not path.is_file() or not 0 < path.stat().st_size <= 16 << 20:
        raise ValueError('测量记录文件须为非空文件且不超过 16 MiB。')
    binding = file_binding(path)
    return {'vertex_indices': [int(first), int(second)], 'length': float(length), 'unit': unit,
            'label': label.strip(), 'correspondence': correspondence.strip(),
            'allowed_absolute_error_mm': float(tolerance),
            'evidence': {'path': str(path), 'sha256': binding['sha256']}}


def render_measurement(run, cad_output):
    import streamlit as st
    from integrations.demo_pipeline import measurement
    key = 'measure_' + hashlib.sha256((str(run) + '\0' + str(cad_output)).encode()).hexdigest()[:16]
    with st.expander('按实测参考长度生成 CAD'):
        st.caption('使用已生成的网格，不重建照片、不重新简化。请在端点预览中选择两个可对应到实物的顶点，并提供实际测量记录。输出单位为毫米；一次定尺度不等于整个物体精度已经验证。')
        if st.button('载入可定尺度网格', key=key + '_load'):
            try:
                st.session_state[key + '_source'] = measurement.inspect_source(run, cad_output)
                st.session_state.pop(key + '_preview', None)
                st.session_state.pop(key + '_result', None)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f'源网格核验失败：{error}')
        source = st.session_state.get(key + '_source')
        if source is None:
            return
        st.caption(f"当前准备网格：{source['vertex_count']} 个顶点、{source['face_count']} 个面；顶点编号从 0 开始，与原始稠密网格编号不同。")
        if st.button('生成端点索引预览', key=key + '_preview_button'):
            try:
                st.session_state[key + '_preview'] = preview(run, cad_output)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f'预览未生成：{error}')
        shown = st.session_state.get(key + '_preview')
        if shown:
            try:
                if shown['mesh'] != source['mesh']:
                    raise ValueError('预览与当前网格不同。')
                st.download_button('下载端点索引查看器', _payload(shown['binding']),
                                   file_name='measurement_vertices.html', mime='text/html', key=key + '_preview_download')
                st.caption('打开后输入顶点编号，确认黄色标记的位置；实际测量的两点必须与这些位置对应。')
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f'预览下载核验失败：{error}')
        with st.form(key + '_form'):
            first = st.number_input('参考段端点 A 编号', 0, source['vertex_count'] - 1, 0, key=key + '_a')
            second = st.number_input('参考段端点 B 编号', 0, source['vertex_count'] - 1, 1, key=key + '_b')
            length = st.text_input('实测参考长度', value='', key=key + '_length')
            unit = st.selectbox('测量单位', ['mm', 'cm', 'm', 'in'], key=key + '_unit')
            label = st.text_input('测量段名称', value='', key=key + '_label')
            correspondence = st.text_area('实物与两个网格端点如何对应', key=key + '_correspondence')
            proof = st.text_input('本机测量记录文件路径', value='', key=key + '_evidence',
                                  help='选择你已有的测量记录或标注文件；系统保存其摘要，不生成虚构测量证据。')
            check_enabled = st.checkbox('增加独立校验尺寸（不参与定尺度）', key=key + '_check_enabled')
            st.caption('以下为可选校验。启用后，测量记录须同时包含该段实测值；超过你填写的误差则不交付成功结果。')
            check_first = st.number_input('校验段端点 A 编号', 0, source['vertex_count'] - 1, 0, key=key + '_check_a')
            check_second = st.number_input('校验段端点 B 编号', 0, source['vertex_count'] - 1, 1, key=key + '_check_b')
            check_length = st.text_input('独立校验实测长度', value='', key=key + '_check_length')
            check_unit = st.selectbox('校验测量单位', ['mm', 'cm', 'm', 'in'], key=key + '_check_unit')
            check_label = st.text_input('校验段名称', value='', key=key + '_check_label')
            check_correspondence = st.text_area('校验段实物与网格端点对应', key=key + '_check_correspondence')
            check_tolerance = st.text_input('校验允许绝对误差（毫米）', value='', key=key + '_check_tolerance')
            submitted = st.form_submit_button('生成带参考尺度的 FCStd / STEP')
        if submitted:
            try:
                if not all(value.strip() for value in (length, label, correspondence, proof)):
                    raise ValueError('请填写实测长度、测量段名称、端点对应说明和测量记录文件。')
                checks = None
                if check_enabled:
                    if {int(check_first), int(check_second)} == {int(first), int(second)}:
                        raise ValueError('独立校验须选择不同于定尺度参考段的端点组合。')
                    checks = [independent_check(check_first, check_second, check_length, check_unit,
                                                check_label, check_correspondence, proof, check_tolerance)]
                with st.spinner('正在定尺度并重新核验 CAD 文件…'):
                    st.session_state[key + '_result'] = execute_reference(
                        run, cad_output, vertex_indices=[int(first), int(second)], length=float(length),
                        unit=unit, label=label, correspondence=correspondence, evidence_path=proof, checks=checks)
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f'定尺度未开始或未完成：{error}')
        saved = st.session_state.get(key + '_result')
        if saved:
            try:
                result = verified_result(saved, run, cad_output)
                if result:
                    st.success('参考尺度 CAD 已生成，单位为毫米；测量记录真实性及整个物体精度尚未验收。')
                    if result.get('independent_check_status') == 'PASS':
                        st.caption('独立校验尺寸符合本次填写的误差阈值；此结论仍依赖测量记录与端点对应的正确性。')
                    else:
                        st.caption('本次未完成独立尺寸校验。')
                    for row in result['artifacts']:
                        suffix = Path(row['path']).suffix
                        st.download_button('下载带尺度 ' + suffix.lstrip('.'), _payload(row),
                                           file_name='measured' + suffix, mime='application/octet-stream', key=key + suffix)
                else:
                    st.warning('定尺度任务未通过；原始模型仍保留。')
                st.caption('定尺度独立任务：' + saved['task'])
            except (OSError, ValueError, KeyError, TypeError) as error:
                st.error(f'定尺度下载核验失败：{error}')
