# 一手来源登记表（provenance registry）

> 本文件 `docs/sources/SOURCES.md` 是来源登记的唯一维护点。任何文档引用 Kozlov 原文时，
> 必须引用本表中的 `id`，不得只写年份。
>
> 最后更新：2026-10-14（既往论文补源；历史T00记录保留）
> `.sources/` 保留 PDF、抽文与短指针，仍被 Git 忽略；普通文档提交不会自动保存这些 PDF/抽文。
> 本地 SHA-256 只标识文件字节，不独立证明来源真实性；历史下载曾关闭证书校验，详见下文。

## 状态图例

| 状态 | 含义 |
|---|---|
| `ARCHIVED` | 本机全文已落盘且哈希已记录；引用仍须核对具体页与适用前提，不表示全文已进入 Git |
| `SOURCE_REQUIRED` | 论文已知存在，但全文不在本仓库，**禁止据标题/摘要补定义** |
| `CROSS_CHECK_ONLY` | 仅作交叉验证，不得作为唯一定义依据 |

---

## S-1996 — `ARCHIVED`

- **id**：`S-1996`
- **原文标题**：Математическое моделирование зрительного восприятия
- **作者**：В. Н. Козлов
- **出处**：Математические вопросы кибернетики, вып. 6. — М.: Наука, 1996. — С. 321–338
- **来源 URL**：`http://library.keldysh.ru/mvk.asp?id=1996-321`（页面首页自带）
- **获取方式**：本仓库既有文件（T00 之前已存在，非本次下载）
- **文件**：[kozlov1996_mvk321.pdf](../../.sources/kozlov1996_mvk321.pdf) / [kozlov1996_mvk321.txt](../../.sources/kozlov1996_mvk321.txt)
- **大小 / SHA-256（pdf）**：2 591 967 B / `1b831bcf92abe1cd638fd3855766e34eafb2846a8a6095ad81b409a40488e3ca`
- **文本层**：有（抽自 pdf，62 477 字符）
- **已知术语**：`проекц`(100)、`аффинн`(39)、`теорем`(21)、`трехмерн`(11)、`стерео`(4)、`метк`(11，实为 *разметка*)、`AlgT'`（正文出现）

## S-2002 — `ARCHIVED`

- **id**：`S-2002`
- **原文标题**：Об одном подходе к сжатию информации при обработке и распознавании изображений
- **作者**：В. Н. Козлов（2026-10-14核首页“© 2002 г.”；旧“Го Во Но”是误录，并非共同作者）
- **出处**：Журнал вычислительной математики и математической физики, 2002, т. 42, № 2, с. 256–268
- **获取方式**：本仓库既有文件（非本次下载）
- **文件**：[kozlov2002_zvmmf1236.pdf](../../.sources/kozlov2002_zvmmf1236.pdf) / [kozlov2002.txt](../../.sources/kozlov2002.txt)
- **大小 / SHA-256（pdf）**：2 484 979 B / `257159fa77b74e1fa503e0a97d6a4b50c3b1123ecfdf8b51cd64d9aff9a831e7`
- **文本层**：有（47 406 字符）
- **已知术语**：`эскиз`(23)、`фрагмент`(20)、`остов`(10)、`эталон`(8)、`круг`(9)、`существенн`(5)、`замкнут`(3)、`дробност`(1)

## S-2009 — `ARCHIVED`（T00 本次取得，此前登记为"仅摘要"）

- **id**：`S-2009`
- **原文标题**：Распознавание изображений, представляемых конечным множеством точек
- **作者**：В. Н. Козлов
- **出处**：Фундаментальная и прикладная математика, 15:5 (2009) 95–110
- **英译**：J. Math. Sci. 172:5 (2011) 690–699，DOI `10.1007/s10958-011-0213-1`
- **来源 URL**：`https://www.mathnet.ru/eng/fpm1248`；全文 `https://www.mathnet.ru/php/getFT.phtml?jrnid=fpm&paperid=1248&what=fullt&option_lang=eng`
- **获取日期**：2026-10-02
- **文件**：[kozlov2009_fpm1248.pdf](../../.sources/kozlov2009_fpm1248.pdf) / [kozlov2009_fpm1248.txt](../../.sources/kozlov2009_fpm1248.txt)
- **大小 / SHA-256（pdf）**：184 724 B / `eb9037306a41a245496672c8e53e7a0b51713df2e2c09cacf4875d77be0a6254`
- **页数**：16（对应 95–110）
- **文本层**：有（43 663 字符）。⚠️ 抽取时 pypdf 报 CFF Type1 字体编码警告（未装 fontTools），
  正文西里尔字母渲染正常，但**数学符号需逐处目视复核**后方可写入公式。
- **已知术语**：`аффинн`(23)、`проекц`(41)、`теорем`(15)、`эскиз`(3)、`стерео`(3)、`существенн`(2)、`метк`(11，疑为 *разметка*)
- **与旧登记的矛盾**：`docs/specs/official_image_code.md` §1 第 5/16 行称"依据 Kozlov 2009 **论文全文**"为 S0/S1，
  但同文件第 76 行记录"2009 论文全文 PDF ❌ 网络路径无法下载（TLS 握手被阻断）；仅取得**摘要**"。
  T00 现已取得全文；本登记说明当前文件状态，旧时间点的获取失败记录保留，后续文档应区分历史与当前状态。

## S-2025 — `ARCHIVED`（T00 本次取得）

- **id**：`S-2025`
- **原文标题**：О трех начальных приближениях к формальному определению визуального образа
  в произвольной визуальной среде
- **作者**：В. Н. Козлов
- **俄文出处**：Интеллектуальные системы. Теория и приложения, 2025, т. 29, вып. 3, с. 146–162
- **英文版**：Pattern Recognition and Image Analysis, 35:4 (2025) 657–664，DOI `10.1134/S1054661825700464`
- **来源 URL**：`https://www.mathnet.ru/eng/ista568`（俄文原文全文入口）
- **获取日期**：2026-10-02
- **文件**：[kozlov2025_three_approximations.pdf](../../.sources/kozlov2025_three_approximations.pdf) / [kozlov2025_three_approximations.txt](../../.sources/kozlov2025_three_approximations.txt)
  （文件名沿用用户指定预留名）
- **大小 / SHA-256（pdf）**：517 327 B / `82fc78d0c0f8fa3de99ede3fd227adaeddc8f8328a1ef99d06755f1ec231b8f9`
- **页数**：17（对应 146–162）
- **语言核验**：正文为**俄文原文**（p.1 标题 «О трех начальных приближениях…»）；
  p.17 附英文标题/摘要与英译参考文献表。
- **文本层**：有（40 838 字符），无字体编码警告
- **已知术语**：`групп`(111)、`круг`(49)、`футляр`(47)、`куск`(38)、`образ`(126)、`остов`(34)、`аффинн`(34)、`полн`(37)、`существенн`(15)、`эллипс`(13)、`замкнут`(13)、`эталон`(12)、`дробност`(11)、`кусок`(10)、`отдельн`(9)、`признак`(4)、`оболочк`(3)、`шлейф`(2)、`метк`(2)、`спектр`(1)

## 降级来源

- **RU2779379C1（专利）** — `CROSS_CHECK_ONLY`。不得作为唯一定义依据。
- **Козлов 2007《Введение в математическую теорию зрительного восприятия》（教材）** —
  T00 未见全文落盘，登记为 `SOURCE_REQUIRED`。2025 论文 [1] 引用它作为"最优互位可由有限过程求得"的依据。

---

## 获取历史与复核边界

历史 T00 下载为处理当时 TLS 连接问题，曾使用 `-k`（关闭证书校验）及 `--ssl-no-revoke`。
这项事实必须保留，但它不是推荐默认值；不能仅由文件 SHA-256 推断下载来源已被认证。
下面默认命令保留正常 TLS 校验；失败时记录原始错误、检查网络/证书/代理，不自动退回关闭校验。
官方入口、元数据、文件头、全文内容和本地哈希分别提供不同证据，不能相互替代。

历史词干统计只帮助定位上下文，不能证明首次、唯一、概念不存在或论文不相关。
本轮 T00-R 目视复核 S-2025 PDF p.2、p.10，并核对 p.13 文本；未完成四篇论文的全面数学审计。
S-2025 p.2 明确提及三维图像和高维推广，后文详细讨论二维；具体新增三维实现仍由项目单独定义和验证。

## 可执行的下载与文本抽取示例（PowerShell）

前提：当前隔离环境的 `python` 可导入 `pypdf`。以下示例使用新的临时目录，避免覆盖原始文件；
下载命令为正常 TLS 校验默认示例，本轮未重新下载。文本抽取主体已用本机 S-2025 PDF 执行复核，
17 个页标记对应 17 页；页码与文本在同一次 `enumerate` 内生成，避免旧示例中 i 超出作用域的问题。

```powershell
$sourceCheckDir = Join-Path $env:TEMP ('kozlov-source-check-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $sourceCheckDir | Out-Null
$pdf = Join-Path $sourceCheckDir 'kozlov2025.pdf'
$txt = Join-Path $sourceCheckDir 'kozlov2025.txt'
curl.exe --fail --location --max-time 120 --retry 2 --output $pdf 'https://www.mathnet.ru/php/getFT.phtml?jrnid=ista&paperid=568&what=fullt&option_lang=eng'
if ($LASTEXITCODE -ne 0) { throw "PDF download failed: $LASTEXITCODE" }

@'
from pathlib import Path
import sys
from pypdf import PdfReader

pdf, txt = map(Path, sys.argv[1:3])
with pdf.open('rb') as stream:
    assert stream.read(5) == b'%PDF-', 'Not a PDF file'
reader = PdfReader(pdf)
page_texts = [
    f'===== page {i} =====\n{page.extract_text() or ""}'
    for i, page in enumerate(reader.pages, start=1)
]
with txt.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write('\n'.join(page_texts))
print(f'Extracted {len(page_texts)} pages')
'@ | python - $pdf $txt
if ($LASTEXITCODE -ne 0) { throw "PDF extraction failed: $LASTEXITCODE" }
Get-FileHash -LiteralPath $pdf -Algorithm SHA256
```

数学符号仍须对照 PDF 目视核对，文本抽取成功不等于公式转写正确。
含控制字符的文本可能被检索工具视作二进制；可用 Python 的 Unicode 文本检索定位，
再核对词形与语境。方法限制见 [theory_evolution.md](../theory_evolution.md)。


## 2026-10-14 既往论文补源（正常TLS，无重试）

原PDF保留在本轮`outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/`，该目录也不因普通代码提交自动归档。完整下载链、UTC、HTTP、字节与原/派生关系见该目录`DOWNLOADS.json`，范围及卷页冲突见`BIBLIOGRAPHY.md`。只核作者/标题/页码不等于已核全部数学公式或已实现算法。新增6篇取得全文，另2篇失败记录保持SOURCE_REQUIRED。

### S-1998-AFFINE — `ARCHIVED`

- **原文标题**：О распознавании аффинно разных дискретных изображений
- **作者**：В. Н. Козлов
- **出处**：1998, 3:3-4, 95-118 (24 PDF pages; rendered final page 118; first page inferred 95). Later references claiming 95-122 conflict.
- **官方来源**：https://intsysmagazine.ru/published/1998/issue_3/parts/part1/03-3-04_kozlov.pdf
- **本地PDF**：[原文](../../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/1998_affine.pdf)；268016 B，24 PDF页
- **SHA-256**：`51901bb7f3cd88dc4da96f178768f26f542a8cc6394642a636a56bf9ed2bead2`
- **抽文**：原始/xHH字形抽文保留；另有CP1251字形解码派生文本，公式仍须目视。末页实际118，后文引用95–122冲突未隐藏。

### S-2014-SYSTEM — `ARCHIVED`

- **原文标题**：Алгоритмы формирования системы взаимосвязанных образов
- **作者**：В. Н. Козлов
- **出处**：2014, 18:2, 99-114
- **官方来源**：https://intsysmagazine.ru/articles/is02/04_kozlov.pdf
- **本地PDF**：[原文](../../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/2014_system.pdf)；678372 B，16 PDF页
- **SHA-256**：`8c24ddac67055a9d5259a153a220b0240d44ec72834e6afb590c455b61bbf51b`

### S-2016-PARTS — `ARCHIVED`

- **原文标题**：Некоторые свойства изображений, составляющих части плоской среды
- **作者**：В. Н. Козлов
- **出处**：2016, 20:3, 155-157
- **官方来源**：https://intsysmagazine.ru/published/2016/issue_3/parts/part1/20-3-26_Kozlov.pdf
- **本地PDF**：[原文](../../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/2016_parts.pdf)；246483 B，3 PDF页
- **SHA-256**：`cd08a817e1a104ea0e2582b538f561bb0b7171700f5eed1ff3f35516b045ff25`

### S-2019-SEGMENTATION — `ARCHIVED`

- **原文标题**：Сегментация изображений и преобразования, сохраняющие форму фигур
- **作者**：В. Н. Козлов
- **出处**：2019, 23:2, 57-70
- **官方来源**：https://intsysmagazine.ru/published/2019/issue_2/23-2.pdf
- **本地PDF**：[原文](../../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/2019_segmentation.pdf)；2001668 B，166 PDF页
- **SHA-256**：`3e46bccbd28613975c4ab65357f5dc25191076423330d2d6bb20cc4250ec66b9`
- **整期与正文**：原文件166页；目标正文PDF页57–70，派生14页文章PDF/抽文另存并绑定原PDF摘要。

### S-2019-CODES — `ARCHIVED`

- **原文标题**：Коды, определяющие изображения с точностью до аффинных преобразований
- **作者**：В. Н. Козлов
- **出处**：2019, 23:3, 57-60
- **官方来源**：https://intsysmagazine.ru/published/2019/issue_3/parts/part1/23-3-04_Kozlov.pdf
- **本地PDF**：[原文](../../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/2019_codes.pdf)；390472 B，4 PDF页
- **SHA-256**：`e688fb172263526787296ed93d0566f890ea9430c6e1bcec3c72c26c786703ff`

### S-2022-VISUAL — `ARCHIVED`

- **原文标题**：Об определении понятия визуального образа
- **作者**：В. Н. Козлов
- **出处**：2022, 26:3, 5-21
- **官方来源**：https://www.mathnet.ru/php/getFT.phtml?jrnid=ista&option_lang=rus&paperid=478&what=fullt
- **本地PDF**：[原文](../../outputs/delegation/KOZLOV_LEGACY/20261014_author_algorithms/sources/2022_visual.pdf)；385321 B，17 PDF页
- **SHA-256**：`a261a04c227e65a428ea83a0f99e8843bd0d22e8593378bbdbcafcb9130c9dff`

### S-1999-VISUAL — `SOURCE_REQUIRED`

- **原文标题**：О зрительном образе, математических подходах к определению этого понятия и о распознавании изображений
- **作者**：В. Н. Козлов
- **出处**：1999, 39:11, 1919-1936
- **官方来源**：https://www.mathnet.ru/php/getFT.phtml?jrnid=zvmmf&paperid=1592&what=fullt&option_lang=rus
- **未取得原因**：HTTP 503；HTTPError: HTTP Error 503: Service Unavailable。禁止从标题补造算法。

### S-2000-SIMILARITY — `SOURCE_REQUIRED`

- **原文标题**：Способ оценки похожести изображений, основанный на преобразованиях подобия
- **作者**：В. Н. Козлов
- **出处**：2000, 40:5, 797-808
- **官方来源**：https://www.mathnet.ru/php/getFT.phtml?jrnid=zvmmf&paperid=1502&what=fullt&option_lang=rus
- **未取得原因**：HTTP 200；TimeoutError: 45 second total request budget exceeded。禁止从标题补造算法。
