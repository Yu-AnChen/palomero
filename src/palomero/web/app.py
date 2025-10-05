import csv
import inspect
import io
import json
import pathlib
import pprint
import shutil
import typing
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime

# ruff: noqa: F405,F403
from fasthtml.common import *

import palomero.models
import palomero.omero_handler

from .config import DB_PATH, PUBLIC_DIR

# ------------------------------------ db ------------------------------------ #
db = database(DB_PATH)
app, rt = fast_app(
    hdrs=[
        Link(rel="stylesheet", href="/css/bootstrap-grid.css"),
        MarkdownJS(),
        # HighlightJS(langs=["python", "javascript", "html", "css"]),
    ],
    title="palomero: Align OMERO ROIs",
    static_path=PUBLIC_DIR,
)


TASK_DIR = PUBLIC_DIR / "assets" / "tasks"


@dataclass
class Project:
    name: str
    description: str
    project_id: str
    created_at: str


@dataclass(kw_only=True)
class AlignmentTask(palomero.models.AlignmentTask):
    created_at: str
    process_start_at: str
    process_end_at: str
    process_status: str
    project_id: str
    alignment_task_id: str


@dataclass
class DefaultTask:
    image_id_from: int = None
    image_id_to: int = None
    channel_from: int = 0
    channel_to: int = 0
    mask_roi_id_from: int = None
    mask_roi_id_to: int = None
    max_pixel_size: float = 20.0
    n_keypoints: int = 10000
    thumbnail_max_size: int = 2000
    sample_size_factor: float = 3.0
    auto_mask: bool = True
    only_affine: bool = False
    map_rois: bool = False
    dry_run: bool = True
    # qc_out_dir: str = ""


def patch_checkbox(data: dict):
    out = {}
    for kk, vv in data.items():
        try:
            if json.dumps(vv) == '["0", "true"]':
                vv = True
        except TypeError:
            pass
        out[kk] = vv
    return out


def empty_str_to_none(data: dict):
    return {kk: None if json.dumps(vv) == '""' else vv for kk, vv in data.items()}


def dataclass_to_input(data_class, default_values: dict = None):
    if default_values is None:
        default_values = {}
    input_type_map = {
        int: "number",
        float: "number",
        bool: "checkbox",
        str: "text",
        typing.Optional[int]: "number",
    }

    annotations = {}
    for cc in inspect.getmro(data_class)[::-1]:
        annotations.update(inspect.get_annotations(cc))

    inputs = []
    for kk, tt in annotations.items():
        input_type = input_type_map[tt]
        value = default_values.get(kk, getattr(data_class, kk, None))
        if tt is bool:
            hidden, checkbox = CheckboxX(
                checked=strtobool(str(value)),
                label=snake2hyphens(kk),
                value="true",
                name=kk,
            )
            hidden.attrs["value"] = "0"
            inputs.append((hidden, checkbox))
            continue
        value = str(value)
        checked = None
        ii = [
            snake2hyphens(kk),
            Input(type=input_type, name=kk, value=value, checked=checked),
        ]
        inputs.append(Label(*ii))
    return inputs


def strtobool(query: str) -> bool:
    if query.lower() in ["y", "yes", "t", "true", "on", "1"]:
        return True
    elif query.lower() in ["n", "no", "f", "false", "off", "0"]:
        return False
    # elif query.lower() in ["null", "none", ""]:
    #     return False
    else:
        print(query)
        raise ValueError


t_project = db.create(Project, pk="project_id")
t_alignment_task = db.create(AlignmentTask, pk="alignment_task_id")


if len(t_project()) == 0:
    ii = 1
    t_project.upsert(
        Project(
            name=f"Example Project {ii} with a long long long long long name",
            description=f"Description of Project {ii}",
            project_id=f"pid-{ii}",
        )
    )

# ---------------------------------------------------------------------------- #
#                                  html parts                                  #
# ---------------------------------------------------------------------------- #


def _app_layout(nav="NAV", side="SIDE", main="MAIN", footer=""):
    loader_css = Style(
        """
        .loader-task {
          width: 1.5rem;
          aspect-ratio: 1;
          border-radius: 50%;
          background: linear-gradient(#0174b0 0 0) top/100% 0% no-repeat #ddd;
          animation: l8 infinite steps(100);
        }
        
        @keyframes l8 {
          100% {
            background-size: 100% 100%;
          }
        }
        """.strip()
    )
    active_link_css = Style(
        """
        :where(a:not([role=button])):is([aria-current]:not([aria-current=false]))::after {
            content: " 👈"
        }
        """.strip()
    )
    return Div(id="app", cls="container-fluid")(
        Header(Nav(nav)),
        Main(cls="row", style="margin-top: 1rem;")(
            loader_css,
            active_link_css,
            Div(id="side", cls="col-3")(side),
            Div(id="main", cls="col-9")(main),
        ),
        Footer(Hr(), Div(id="footer", cls="row")(_footer())),
    )


def _footer():
    import palomero

    return (
        Div(cls="row justify-content-center row-cols-auto align-items-center")(
            Div(cls="col")("Task Manager"),
            Div(
                cls="loader-task col",
                hx_get=manage_task.to(),
                hx_swap="outerHTML",
                hx_trigger="every 2s",
                style="animation-duration: 2s;",
            ),
            Div(cls="col gx-5"),
            Div(cls="col")(
                "OMERO",
                hx_get=check_omero_connection.to(),
                hx_trigger="load,every 1000m",
                hx_target="#omero-indicator",
                hx_sync="this:drop",
                hx_preserve=True,
            ),
            Div(
                cls="col", id="omero-indicator", style="padding: 0px; font-size: 1.5rem"
            )("❓"),
        ),
        Div(Small(f"palomero v{palomero.__version__}"), style="text-align: center;"),
    )


def _nav(project_id: str = None, is_new_project: bool = False):
    text_branding = Hgroup(
        hx_trigger="click",
        hx_get="/",
        hx_target="#app",
        hx_swap="outerHTML",
        hx_push_url="true",
        style="cursor: pointer;",
    )(
        H3("palomero"),
        Small("Align OMERO ROIs"),
    )
    projects = [
        A(
            pp.name,
            link=uri("get_task", project_id=pp.project_id, alignment_task_id="new"),
            aria_current=pp.project_id == project_id,
        )
        for pp in t_project()
    ]
    load_project = Details(cls="dropdown")(
        Summary(
            "Switch Project"
            if project_id not in t_project
            else t_project[project_id].name,
            role="button",
            cls="outline contrast",
        ),
        Ul(dir="ltr")(map(lambda x: Li(x(cls="contrast")), projects)),
    )
    delete_confirm = ""
    if project_id in t_project:
        name = t_project[project_id].name
        tasks = t_alignment_task("project_id=?", (project_id,))
        delete_confirm = f"Delete Project '{name}' and all its tasks ({len(tasks)})?"
    manage_project = Div(role="group", style="margin: 0px;")(
        Button(
            "Project:",
            cls="outline contrast",
            disabled=True,
        ),
        Button(
            "Delete",
            hx_post=init_delete_project.to(project_id=project_id),
            hx_confirm=delete_confirm,
            hx_target="#main",
            hx_swap="afterend",
            cls="secondary",
            disabled=project_id not in t_project,
        ),
        Button(
            "New",
            hx_get=get_project.to(project_id="new"),
            hx_target="#app",
            hx_swap="outerHTML",
            hx_push_url="true",
            disabled=is_new_project,
        ),
    )
    return map(Ul, map(Li, [text_branding, load_project, manage_project]))


def _form_project():
    return Form(
        H2("Create New Project"),
        Input(
            type="text",
            name="name",
            placeholder="Name",
            aria_label="name",
            required=True,
        ),
        Input(type="hidden", name="project_id", value=uuid.uuid4()),
        Input(
            type="text",
            name="description",
            placeholder="Description",
            aria_label="description",
        ),
        Input(type="submit", value="Create"),
        hx_post=post_project.to(),
        hx_target="#app",
        hx_swap="outerHTML",
    )


def _side_task_link(alignment_task: AlignmentTask, aria_current: bool = False):
    text = f"{alignment_task.image_id_from} → {alignment_task.image_id_to}"
    now = datetime.fromisoformat(alignment_task.created_at)
    time = datetime.strftime(now, "%Y-%m-%d %H:%M")

    status_text = "⏸️ "
    is_running = False
    if alignment_task.process_start_at:
        is_running = True
        status_text = ""
    if alignment_task.process_end_at:
        is_running = False
        status_text = "✅ "

    return Section(cls="offset-1")(
        A(
            text,
            link=uri(
                "get_task",
                project_id=alignment_task.project_id,
                alignment_task_id=alignment_task.alignment_task_id,
            ),
            aria_current=aria_current,
        ),
        Br(),
        Small(
            status_text,
            time,
            aria_busy=str(is_running).lower(),
            id=f"status-task-{alignment_task.alignment_task_id}",
        ),
    )


def _side_task_section(project_id: str, alignment_task_id: str):
    tasks = [
        _side_task_link(
            tt,
            aria_current=True if tt.alignment_task_id == alignment_task_id else False,
        )
        for tt in t_alignment_task("project_id=?", (project_id,), order_by="created_at")
    ]
    return Div(id=f"section-results-{project_id}")(*tasks[::-1])


def _side(project_id: str = None, alignment_task_id: str = None):
    return (
        Section(
            H5("Run"),
            Div(cls="offset-1")(
                A(
                    "Pair",
                    link=uri(
                        "get_task", project_id=project_id, alignment_task_id="new"
                    ),
                ),
            ),
            Div(cls="offset-1")(
                A(
                    "Batch",
                    link=uri(
                        "get_task", project_id=project_id, alignment_task_id="new_batch"
                    ),
                )
            ),
        ),
        Section(
            H5("Results"),
            _side_task_section(project_id, alignment_task_id),
            style="margin-top: 2.5rem;",
        ),
    )


def _form_pair(project_id: str, data: dict = None):
    # print("data:", data, "\n\n\n")
    inputs = dataclass_to_input(DefaultTask, data)
    for ii in inputs:
        if "image_id" in str(ii):
            ii[1].required = True
    inputs.extend(
        [
            Input(type="hidden", name="alignment_task_id", value=uuid.uuid4()),
            Input(type="hidden", name="project_id", value=project_id),
        ]
    )
    return Form(
        H2("Align One Pair of Images"),
        Section(cls="row")(
            # this div is needed for placing the checkbox input,
            map(lambda x: Div(cls="col-6")(x), inputs),
        ),
        Section(Input(type="submit", value="Run Pair")),
        hx_post=add_pair_alignment_task.to(),
        hx_swap="afterend",
    )


def _dicts_to_table(dicts: list, index: bool = True):
    # FIXME: should handle list of dictionaries with different keys
    header_row = Tr(map(lambda x: Th(x, scope="col"), dicts[0].keys()))

    rows = []
    for ii, dd in enumerate(dicts):
        row = Tr(map(lambda x: Th(x, scope="row") if ii == 0 else Td(x), dd.values()))
        if not index:
            row = Tr(map(lambda x: Td(x), dd.values()))
        rows.append(row)
    return Table(Thead(header_row), Tbody(*rows))


def _form_batch(project_id: str, data: dict = None):
    if data is None:
        data = {}
    data = patch_checkbox(data)
    inputs = [
        ii
        for ii in dataclass_to_input(DefaultTask, data)
        if ("image_id" not in str(ii)) & ("mask_roi_id" not in str(ii))
    ]
    inputs.extend(
        [
            Input(type="hidden", name="alignment_task_id", value=uuid.uuid4()),
            Input(type="hidden", name="project_id", value=project_id),
        ]
    )
    has_valid_csv = len(data.get("tasks", [])) > 0
    table = Table()
    input_data = Input(type="hidden", name="tasks", id="tasks")
    if has_valid_csv:
        table = (
            H4(f"File: {data['csv_file'].filename}"),
            _dicts_to_table(data["tasks"], index=True),
        )
        table = Details(
            Summary(role="button", cls="outline secondary")(
                f"File: {data['csv_file'].filename}"
            ),
            Div(cls="overflow-auto")(_dicts_to_table(data["tasks"], index=True)),
            open=True,
        )
        input_data.value = json.dumps(data["tasks"])

    form = Form(
        H2("Batch Processing from CSV"),
        hx_post=validate_csv.to(),
        # hx_target="#table-from-csv",
        hx_trigger="input[event.target.matches('input[type=\"file\"]')] from:body",
        hx_encoding="multipart/form-data",
        hx_swap="outerHTML",
    )(
        Section(
            Input(type="file", name="csv_file", accept=".csv", value=None),
            # Div(id="table-from-csv")(table),
        ),
        Section(id="table-from-csv")(table, input_data),
        Section(cls="row")(
            # this div is needed for placing the checkbox input,
            map(lambda x: Div(cls="col-6")(x), inputs),
        ),
        Section(
            Input(
                type="submit",
                value="Run",
                hx_post=add_batch_alignment_tasks.to(),
                hx_swap="afterend",
                disabled=not has_valid_csv,
            )
        ),
    )

    return form


def _parse_tqdm_log(file_path: str) -> str:
    output_lines = ["\n"]
    try:
        with open(file_path, "r") as f:
            for line in f:
                last = output_lines[-1]
                if line == "\n":
                    if "\x1b[A\n" in last:
                        continue
                if ("\x1b[A\n" in line) & ("\x1b[A\n" in last):
                    output_lines[-1] = line
                    continue
                output_lines.append(line)

        return "".join(output_lines[1:]).replace("\x1b[A", "")
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"An error occurred: {e}"


def _task_log(alignment_task_id: str):
    task = t_alignment_task[alignment_task_id]
    _, runnings = query_tasks(task.project_id)
    is_running = alignment_task_id in [rr.alignment_task_id for rr in runnings]

    qc_dir = TASK_DIR / task.project_id / task.alignment_task_id

    if not qc_dir.exists():
        return

    if task.process_start_at is None:
        return

    p_logs = sorted(qc_dir.glob("*.log"))
    p_jpgs = sorted(qc_dir.glob("*.jpg"))

    log_text: str = ""
    if p_logs:
        log_text = _parse_tqdm_log(p_logs[0])
    log = Pre(Code(log_text, data_highlighted="yes"), style="font-size: 10px;")

    img = []
    if p_jpgs:
        for pp in p_jpgs:
            pp = pp.relative_to(PUBLIC_DIR)
            img.append(Figure(A(Img(src=f"/{pp}"), href=f"/{pp}"), Figcaption(pp.name)))

    aria_busy: bool = True if is_running else False
    detail_open: bool = True if is_running else False
    hx_kwargs = {}
    if is_running:
        # timed refresh the app for log update
        hx_kwargs = dict(
            hx_get=get_task.to(
                project_id=task.project_id,
                alignment_task_id=task.alignment_task_id,
            ),
            hx_trigger="every 2s",
            hx_target="main",
            hx_select="main",
            hx_swap="outerHTML",
            hx_preserve=True,
        )

    return Details(
        Summary(
            "Log",
            aria_busy=str(aria_busy).lower(),
            role="button",
            cls="secondary outline",
        ),
        log,
        open=detail_open,
        **hx_kwargs,
    ), *img


def _alignment_task_result(alignment_task_id: str):
    alignment_task = t_alignment_task[alignment_task_id]
    headding = H2(
        f"Task: {alignment_task.image_id_from} → {alignment_task.image_id_to}"
    )
    task_setting = Details(
        Summary("Run Settings", role="button", cls="secondary outline"),
        Pre(Code(pprint.pformat(alignment_task, sort_dicts=False))),
        open=False,
    )
    task_log = _task_log(alignment_task_id)
    delete_button = Button(
        "Delete task",
        cls="secondary",
        hx_confirm="Delete current task?",
        hx_post=delete_alignment_task.to(alignment_task_id=alignment_task_id),
    )
    run_button = A(
        "Run again",
        role="button",
        hx_get=get_task.to(
            project_id=alignment_task.project_id, alignment_task_id="new"
        ),
        # NOTE: passing dataclass does not work!
        hx_vals=asdict(alignment_task),
        hx_swap="outerHTML",
        hx_target="main",
        hx_select="main",
        hx_push_url="true",
    )
    return (
        headding,
        task_setting,
        task_log,
        Div(cls="grid")(delete_button, run_button),
    )


def _submitted_modal(data, mode, task_links):
    import copy

    # FIXME the modal should trigger manage_task!
    assert mode in ["pair", "batch"]
    project_id = data["project_id"]

    pairs = [copy.deepcopy(ll[0]) for ll in task_links]
    for pp in pairs:
        if "class" in pp.attrs:
            del pp.attrs["class"]

    task_id = "new_batch" if mode == "batch" else "new"
    hx_kwargs = dict(
        hx_get=get_task.to(
            project_id=project_id,
            alignment_task_id=task_id,
        ),
        hx_swap="outerHTML",
        hx_target="#app",
        hx_push_url="true",
    )
    buttons = [
        Button("Close", cls="secondary", hx_vals=data, **hx_kwargs),
        Button(f"New {mode.capitalize()}", **hx_kwargs),
    ]
    title = f"Submitted {len(task_links)} task{'s' if len(task_links) > 1 else ''}"
    dialog = Dialog(open=True, id="submitted-dialog")(
        Article(
            Nav(
                Ul(Li(H3(title, style="margin: 1px;"))),
                Ul(map(Li, buttons)),
            ),
            Hr(),
            Div(cls="row")(*[Div(cls="col")(pp) for pp in pairs]),
        )
    )
    return dialog


# ---------------------------------------------------------------------------- #
#                                    routes                                    #
# ---------------------------------------------------------------------------- #
@rt("/project/{project_id}/task/{alignment_task_id}")
def get_task(project_id: str = None, alignment_task_id: str = None, data: dict = None):
    if (alignment_task_id is None) or (alignment_task_id == ""):
        return _app_layout(
            nav=_nav(project_id=project_id),
            main="LIST OF TASKS of this project",
            side="",
        )
    if alignment_task_id == "new":
        return _app_layout(
            nav=_nav(project_id=project_id),
            main=_form_pair(project_id, data=data),
            side=_side(project_id, alignment_task_id),
        )
    if alignment_task_id == "new_batch":
        return _app_layout(
            nav=_nav(project_id=project_id),
            main=_form_batch(project_id=project_id, data=data),
            side=_side(project_id, alignment_task_id),
        )
    if project_id not in t_project:
        return Redirect("/")
    return _app_layout(
        nav=_nav(project_id=project_id),
        main=_alignment_task_result(alignment_task_id=alignment_task_id),
        side=_side(project_id, alignment_task_id),
    )


@rt("/project/{project_id}")
def get_project(project_id: str = None):
    if (project_id is None) or (project_id == ""):
        return _app_layout(
            nav=_nav(None),
            main="LIST OF PROJECTS?",
            side="",
        )
    if project_id == "new":
        return _app_layout(
            nav=_nav(None, is_new_project=True),
            main=_form_project(),
            side="",
        )
    if project_id not in t_project:
        return Redirect("/")
    return _app_layout(
        nav=_nav(project_id),
        main=project_id,
        side=_side(project_id),
    )


@rt
def post_project(project_id: str, name: str, description: str):
    project = Project(
        name=name,
        description=description,
        project_id=project_id,
        created_at=datetime.now().isoformat(),
    )
    project = t_project.upsert(project)
    return Redirect(f"/project/{project.project_id}/task/new")


@rt
def index(project_id: str = None, alignment_task_id: str = None):
    text = open(PUBLIC_DIR / "TUTORIAL.md", encoding="utf-8").read()
    main = Div(text, cls="marked")
    if (project_id is None) and (alignment_task_id is None):
        return _app_layout(nav=_nav(), main=main, side="")


@rt
async def validate_csv(data: dict):
    csv_file = data.get("csv_file")
    content = await csv_file.read()
    csv_file = io.StringIO(content.decode("utf-8-sig"))
    reader = csv.DictReader(csv_file)
    tasks = []
    for ii, row in enumerate(reader, 1):
        if not (bool(row["image-id-from"]) & bool(row["image-id-to"])):
            continue
        # NOTE: no value to None
        row = {kk: None if vv == "" else vv for kk, vv in row.items()}
        task = {"row_num": ii}
        for kk in inspect.get_annotations(DefaultTask).keys():
            tk = kk.replace("_", "-")
            if tk in row:
                task.update({kk: row[tk]})
        tasks.append(task)
    data.update({"tasks": tasks})
    return _form_batch(project_id=data["project_id"], data=data)


@rt
def add_pair_alignment_task(data: dict):
    data = patch_checkbox(data)
    task_data = empty_str_to_none(data)
    task = AlignmentTask(**task_data)
    now = datetime.now()
    task.created_at = now.isoformat()

    task = t_alignment_task.upsert(task)

    task_link = Section(
        id=f"section-results-{task.project_id}", hx_swap_oob="afterbegin"
    )(_side_task_link(alignment_task=task, aria_current=False))

    return (
        _submitted_modal(task_data, "pair", [task_link]),
        task_link,
        manage_task(task.project_id)[0](hx_swap_oob="outerHTML"),
    )


@rt
def add_batch_alignment_tasks(data: dict):
    data = patch_checkbox(data)
    _tasks = json.loads(data["tasks"])
    del data["tasks"]

    tasks = []
    for tt in _tasks:
        tt["alignment_task_id"] = str(uuid.uuid4())
        for kk, vv in data.items():
            if tt.get(kk, None) is None:
                tt.update({kk: vv})
        tasks.append(add_pair_alignment_task(tt)[1])

    return (
        _submitted_modal(data, "batch", tasks),
        *tasks,
        manage_task(tasks[0].project_id)[0](hx_swap_oob="outerHTML"),
    )


@rt
def delete_alignment_task(alignment_task_id: str):
    task = t_alignment_task[alignment_task_id]
    t_alignment_task.delete(alignment_task_id)

    qc_dir = TASK_DIR / task.project_id / task.alignment_task_id
    if qc_dir.exists():
        shutil.rmtree(qc_dir)
    broadcast_alignment_task_status()

    return Redirect(get_task.to(project_id=task.project_id, alignment_task_id="new"))


@rt
def delete_project(project_id: str):
    tasks = t_alignment_task("project_id=?", (project_id,))
    for tt in tasks:
        _ = delete_alignment_task(tt.alignment_task_id)
    t_project.delete(project_id)
    return Redirect("/")


@rt
def init_delete_project(project_id: str = None):
    return Dialog(
        aria_busy="true",
        open=True,
        hx_trigger="load once",
        hx_post=delete_project.to(project_id=project_id),
    )


@rt
def check_omero_connection():
    if palomero.omero_handler.get_omero_connection() is None:
        return "⛔"
    return "⚡️"


# -------------------------------- task queue -------------------------------- #
def query_tasks(project_id: str = None, max_runtime: float = None):
    """return (<tasks that are not launched>, <tasks currently running>)"""
    MAX_RUNTIME = 10 * 60
    if max_runtime is None:
        max_runtime = MAX_RUNTIME

    projects = t_project(order_by="created_at")[::-1]
    if project_id in t_project:
        current_project = projects.pop(
            [pp.project_id for pp in projects].index(project_id)
        )
        projects.insert(0, current_project)
    tasks = [
        t_alignment_task(
            "project_id = ? AND process_start_at IS NULL",
            [pp.project_id],
            order_by="created_at",
        )
        for pp in projects
    ]
    tasks = flat_xt(tasks)

    runnings = []
    _runnings = t_alignment_task(
        "process_start_at IS NOT NULL AND process_end_at IS NULL"
    )
    for rr in _runnings:
        diff = (
            datetime.now() - datetime.fromisoformat(rr.process_start_at)
        ).total_seconds()
        if diff <= max_runtime:
            runnings.append(rr)
        else:
            status = (
                f"Exceed max runtime ({max_runtime / 60:.1f}) minutes - process killed"
            )
            rr.process_end_at = datetime.now().isoformat()
            rr.process_status = status
            t_alignment_task.upsert(rr)
    return tasks, runnings


@threaded
def launch_task(task: AlignmentTask):
    import subprocess

    qc_dir = TASK_DIR / task.project_id / task.alignment_task_id
    qc_dir.mkdir(exist_ok=True, parents=True)

    cmd = ["palomero"]
    for kk, vv in inspect.get_annotations(DefaultTask).items():
        cli_key = f"--{kk.replace('_', '-')}"
        cli_value = getattr(task, kk)
        if vv is bool:
            if strtobool(str(cli_value)):
                cmd.append(cli_key)
            continue
        if cli_value is None:
            continue
        cmd.extend([cli_key, str(cli_value)])
    cmd.extend(["--qc-out-dir", qc_dir])
    # pprint.pprint(cmd)

    task.process_start_at = datetime.now().isoformat()
    t_alignment_task.update(task)
    popen = subprocess.Popen(
        cmd,
        stderr=subprocess.STDOUT,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    with open(qc_dir / "palomero.log", "w"):
        ...
    for ll in popen.stdout:
        with open(qc_dir / "palomero.log", "a") as fout:
            fout.write(ll)
    popen.stdout.close()
    return_code = popen.wait()
    print(task.alignment_task_id, "return code", return_code)
    task.process_end_at = datetime.now().isoformat()
    return t_alignment_task.update(task)


@rt
def manage_task(project_id: str = None):
    tasks, runnings = query_tasks(project_id)
    updates = broadcast_alignment_task_status()
    manager = Div(
        id="task-manager",
        cls="loader-task col",
        hx_get=manage_task.to(),
        hx_swap="outerHTML",
    )
    if runnings:
        return manager(hx_trigger="every 2s", style="animation-duration: 2s;"), *updates
    if tasks:
        launch_task(tasks[0])
        broadcast_alignment_task_status()
        return manager(hx_trigger="every 2s", style="animation-duration: 2s;"), *updates
    return manager(hx_trigger="every 300s", style="animation-duration: 300s;"), *updates


def broadcast_alignment_task_status():
    projects = t_project()
    updates = []
    for pp in projects:
        project_tasks = _side_task_section(
            project_id=pp.project_id, alignment_task_id=""
        )
        for tt in project_tasks.children:
            updates.extend(list(filter(lambda x: x.tag == "small", tt.children)))

    for uu in updates:
        uu.hx_swap_oob = "outerHTML"
    return updates


def run():
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Launch palomero web-app")
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to launch the server on (default: 5001)",
    )
    args = parser.parse_args()

    curr = pathlib.Path(__file__).parent
    os.chdir(curr)
    serve(appname="palomero.web.app", host="localhost", port=args.port, reload=True)


# serve()
