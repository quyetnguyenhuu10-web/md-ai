from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def text(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_chat_redundant_files_removed():
    assert not (REPO / "md").exists()
    assert not (REPO / "apps" / "terminal" / "lib" / "color.ts").exists()
    assert not (
        REPO / "apps" / "terminal" / "components" / "mintdim_anglerfish.ts"
    ).exists()


def test_chat_worker_cli_owns_default_checkpoint_resolution():
    body = text("src/mintdim_lab/serving/worker_cli.py")

    assert "def default_checkpoint()" in body
    assert "CHAT_CHECKPOINT_LIST_PATH" in body
    assert "resolve_repo_path" in body
    assert "from .checkpoints import default_checkpoint" not in body


def test_inference_server_no_longer_imports_worker_checkpoint_module():
    body = text("src/mintdim_lab/serving/http_api.py")
    command = text("src/mintdim_lab/cli/commands/serve.py")

    assert "md.chat.worker.checkpoints" not in body
    assert "def _default_checkpoint()" in body
    assert "CHAT_CHECKPOINT_LIST_PATH" in body
    assert "def build_parser()" in command
    assert "run_server(" in command


def test_theme_owns_color_helpers():
    body = text("apps/terminal/lib/theme.ts")

    assert 'from "./color"' not in body
    assert "function rgbToHex(" in body
    assert "function lerpRgbToHex(" in body
    assert 'export const ICON_COLOR = "#5cdcb4";' in body
    assert 'export const RESPONDING_LABEL_COLOR = "#5cdcb4";' in body
    assert "const TEXT_BASE_RGB: readonly [number, number, number] = [92, 220, 180];" in body
    assert "const TEXT_HIGHLIGHT_RGB: readonly [number, number, number] = [170, 245, 220];" in body


def test_chat_python_module_entrypoint_is_worker_only():
    package_json = text("apps/terminal/package.json")
    worker = text("apps/terminal/lib/worker.ts")
    legacy_worker_command = "python -m " + "md" + ".chat.worker"
    legacy_worker_module = '"' + "md" + ".chat.worker" + '"'

    assert "python ../../src/mintdim_lab/cli/main.py chat --worker" in package_json
    assert '"src/mintdim_lab/cli/main.py", "chat", "--worker"' in worker
    assert legacy_worker_command not in package_json
    assert legacy_worker_module not in worker


def test_top_level_cli_dispatches_chat_command():
    body = text("src/mintdim_lab/cli/main.py")

    assert 'if command == "chat":' in body
    assert "return chat.main(rest)" in body


def test_chat_command_launches_ui_or_worker():
    body = text("src/mintdim_lab/cli/commands/chat.py")

    assert 'prog=command_prog("chat")' in body
    assert 'app_dir = repo / "apps" / "terminal"' in body
    assert 'script = app_dir / "chat.tsx"' in body
    assert "subprocess.call(cmd, cwd=app_dir)" in body
    assert "from mintdim_lab.serving.worker_cli import main as worker_main" in body


def test_chat_ui_repo_root_stays_at_repository_root():
    body = text("apps/terminal/App.tsx")

    assert "const HERE = import.meta.dir;" in body
    assert 'const REPO_ROOT = join(HERE, "..", "..");' in body
    assert 'const CHECKPOINT_YAML = join(REPO_ROOT, "recipes", "chat", "checkpoints.yaml");' in body
    assert 'const REPO_ROOT = join(HERE, "..", "..", "..", "..");' not in body


def test_chat_ui_uses_public_python_default():
    body = text("apps/terminal/App.tsx")

    assert 'let python = process.env.MINTDIM_PYTHON ?? "python";' in body
    assert "process.platform" not in body
    assert '? "py" : "python3"' not in body


def test_chat_ui_uses_split_footer_with_retained_scrollback():
    launcher = text("apps/terminal/chat.tsx")
    app = text("apps/terminal/App.tsx")

    assert 'from "./lib/anglerfish"' in launcher
    assert "drawStartupScreen();" in launcher
    assert 'const CLEAR_VISIBLE_SCREEN = "\\x1b[2J";' in launcher
    assert 'const HOME = "\\x1b[H";' in launcher
    assert 'screenMode: "split-footer"' in launcher
    assert 'externalOutputMode: "capture-stdout"' in launcher
    assert "footerHeight: 3" in launcher
    assert "clearOnShutdown: false" in launcher
    assert "const MESSAGE_MARGIN_X = 2;" in app
    assert "const IDLE_FOOTER_HEIGHT = 3;" in app
    assert "const ACTIVE_FOOTER_HEIGHT = 4;" in app
    assert "writeSolidToScrollback" not in app
    assert "async function appendUserMessage(text: string)" in app
    assert "BoxRenderable" in app
    assert "TextRenderable" in app
    assert "<StatusLine" in app
    assert "<InputBar" in app
    assert "process.stdout.write" not in app


def test_chat_anglerfish_banner_is_import_safe():
    body = text("apps/terminal/lib/anglerfish.ts")

    assert "export function renderAnglerfishBanner" in body
    assert "export function renderAnglerfishFrame" in body
    assert "if (isDirectRun())" in body
    assert "void main();" in body


def test_chat_ui_commits_bot_markdown_with_retained_scrollback_surface():
    app = text("apps/terminal/App.tsx")

    assert "MarkdownRenderable" in app
    assert "renderer.createScrollbackSurface" in app
    assert "const contentWidth = Math.max(1, surface.width - MESSAGE_MARGIN_X * 2);" in app
    assert "left: MESSAGE_MARGIN_X" in app
    assert "width: contentWidth" in app
    assert "await surface.settle();" in app
    assert "surface.commitRows(0, surface.height" in app
    assert 'if (role === "bot")' in app


def test_chat_ui_keeps_status_tight_above_input():
    app = text("apps/terminal/App.tsx")

    status_index = app.index("<StatusLine")
    spacer_index = app.index("<box flexGrow={1} />")
    input_index = app.index("<InputBar")

    assert status_index < spacer_index < input_index
    assert "const ACTIVE_FOOTER_HEIGHT = 4;" in app
    assert '<Show when={phase() !== "idle"}>' in app
    assert "<box flexShrink={0} height={1}>" in app
    assert "<box flexShrink={0} height={1} paddingX={MESSAGE_MARGIN_X}>" not in app
    assert "renderer.footerHeight = next;" in app


def test_chat_user_bubble_starts_at_terminal_left_edge():
    app = text("apps/terminal/App.tsx")
    user_block = app[
        app.index("async function appendUserMessage") : app.index("async function appendMessage")
    ]

    assert "const bubbleWidth = Math.max(1, surface.width);" in user_block
    assert "left: 0" in user_block
    assert "width: bubbleWidth" in user_block
    assert "content: `❯ ${text}`" in user_block
    assert "backgroundColor: USER_BUBBLE_BG" in user_block
    assert "paddingLeft" not in user_block
    assert "paddingRight" not in user_block
    assert "bg: USER_BUBBLE_BG" in user_block
    assert "paddingBottom={1}" not in app


def test_chat_ui_flushes_response_before_clearing_live_status():
    app = text("apps/terminal/App.tsx")

    stream_start = app.index("sendPrompt(text, {")
    done_block = app[
        app.index("onDone: () => {", stream_start) : app.index(
            "resolve();", app.index("onDone: () => {", stream_start)
        )
    ]

    assert 'await appendMessage("bot", final);' in done_block
    assert "stopStatus();" in done_block
    assert done_block.index('await appendMessage("bot", final);') < done_block.index(
        "stopStatus();"
    )
