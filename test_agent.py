#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модульные тесты для agent.py (запуск: pytest test_agent.py -v).

Проверяются:
  * truncate — ограничение текста 20 символами;
  * load_config — чтение api_key/base_url из YAML, обработка ошибок;
  * EXAMPLE_CONFIG — пример конфигурации валиден и содержит обязательные ключи;
  * log_api / log_tool — формат записей в логах;
  * save_history — сохранение истории чата в отдельный файл (UTF-8, JSONL);
  * run_shell — полноценный shell (stdout/stderr/returncode, таймаут, журнал);
  * SYSTEM_PROMPT — запрет нечитаемых символов UTF-8;
  * RUN_SHELL_TOOL — описание TOOL (function_call) для передачи через API;
  * main() — интеграция: цикл чата с мок-API, tool_calls, логирование.
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent


# ---------------------------------------------------------------- fixtures ---

@pytest.fixture(autouse=True)
def redirect_logs(tmp_path, monkeypatch):
    """Перенаправляем все файлы вывода (логи/история) во временную директорию."""
    monkeypatch.setattr(agent, "API_LOG_PATH", str(tmp_path / "api.log"))
    monkeypatch.setattr(agent, "TOOL_LOG_PATH", str(tmp_path / "tool_usage.log"))
    monkeypatch.setattr(agent, "HISTORY_PATH", str(tmp_path / "chat_history.jsonl"))
    yield


# ------------------------------------------------------------------ truncate --

class TestTruncate:
    def test_default_limit_20_chars(self):
        s = "Привет, это очень длинное сообщение для проверки"
        assert len(agent.truncate(s)) == 20
        assert agent.truncate(s) == s[:20]

    def test_short_text_unchanged(self):
        assert agent.truncate("команда") == "команда"

    def test_newlines_replaced_with_space(self):
        assert agent.truncate("строка1\nстрока2") == "строка1 строка2"

    def test_none_and_empty(self):
        assert agent.truncate(None) == ""
        assert agent.truncate("") == ""

    def test_custom_limit(self):
        assert agent.truncate("abcdefghij", 4) == "abcd"


# --------------------------------------------------------------- load_config --

class TestLoadConfig:
    def test_valid_config(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text(
            'api_key: "sk-test-key"\nbase_url: "http://localhost:8000/v1"\nmodel: "test-model"\n',
            encoding="utf-8",
        )
        cfg = agent.load_config(str(p))
        assert cfg["api_key"] == "sk-test-key"
        assert cfg["base_url"] == "http://localhost:8000/v1"
        assert cfg["model"] == "test-model"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(agent.ConfigError) as exc:
            agent.load_config(str(tmp_path / "absent.yaml"))
        # сообщение должно содержать пример конфигурации
        assert "api_key" in str(exc.value)

    def test_missing_required_keys(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text('base_url: "http://x/v1"\n', encoding="utf-8")
        with pytest.raises(agent.ConfigError, match="api_key"):
            agent.load_config(str(p))

        p2 = tmp_path / "config2.yaml"
        p2.write_text('api_key: "sk"\n', encoding="utf-8")
        with pytest.raises(agent.ConfigError, match="base_url"):
            agent.load_config(str(p2))

    def test_not_a_mapping(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text("- просто список\n", encoding="utf-8")
        with pytest.raises(agent.ConfigError):
            agent.load_config(str(p))

    def test_example_config_is_valid_yaml_with_required_keys(self):
        cfg = __import__("yaml").safe_load(agent.EXAMPLE_CONFIG)
        assert isinstance(cfg, dict)
        assert "api_key" in cfg and "base_url" in cfg


# -------------------------------------------------------------- validate_config --

class TestValidateConfig:
    def test_swapped_model_and_url_rejected(self):
        """Пользователь перепутал model и base_url -> понятная ошибка, не трейсбек."""
        cfg = {
            "api_key": "sk",
            "model": "http://localhost:8000/v1",
            "base_url": "gpt-4o-mini",
        }
        with pytest.raises(agent.ConfigError, match="перепутаны"):
            agent.validate_config(cfg)

    def test_non_http_base_url_rejected(self):
        with pytest.raises(agent.ConfigError, match="http"):
            agent.validate_config({"api_key": "sk", "base_url": "ftp://host/v1"})

    def test_base_url_without_scheme_rejected(self):
        with pytest.raises(agent.ConfigError):
            agent.validate_config({"api_key": "sk", "base_url": "localhost:8000/v1"})

    def test_url_in_model_rejected(self):
        with pytest.raises(agent.ConfigError, match="model"):
            agent.validate_config({
                "api_key": "sk",
                "base_url": "http://localhost:8000/v1",
                "model": "http://gpt-4o-mini",
            })

    def test_valid_config_passes(self):
        agent.validate_config({
            "api_key": "sk",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
        })

    def test_missing_model_defaults_ok(self):
        agent.validate_config({"api_key": "sk", "base_url": "http://localhost:8000/v1"})

    def test_load_config_runs_validation(self, tmp_path):
        p = tmp_path / "config.yaml"
        p.write_text('api_key: "sk"\nmodel: "http://localhost:8000/v1"\nbase_url: "gpt-4o-mini"\n', encoding="utf-8")
        with pytest.raises(agent.ConfigError, match="перепутаны"):
            agent.load_config(str(p))


# ------------------------------------------------------------------- log_api --

class TestLogApi:
    def test_format_and_truncation(self):
        long_text = "Сообщение пользователя, которое точно длиннее двадцати символов"
        agent.log_api("user", long_text)
        line = open(agent.API_LOG_PATH, encoding="utf-8").read().strip()
        # [timestamp] kind: <не более 20 символов>
        ts, rest = line.split("] ", 1)
        assert ts.startswith("[")
        kind, payload = rest.split(": ", 1)
        assert kind == "user"
        assert len(payload) <= 20
        assert payload == long_text[:20]

    def test_model_kind_logged(self):
        agent.log_api("model", "ответ модели")
        content = open(agent.API_LOG_PATH, encoding="utf-8").read()
        assert "model: ответ модели" in content

    def test_appends_multiple_lines(self):
        agent.log_api("user", "раз")
        agent.log_api("model", "два")
        lines = open(agent.API_LOG_PATH, encoding="utf-8").read().splitlines()
        assert len(lines) == 2


# ------------------------------------------------------------------ log_tool --

class TestLogTool:
    def test_fields_present(self):
        agent.log_tool("run_shell", {"command": "ls -la"}, '{"rc": 0}', rc=0)
        line = open(agent.TOOL_LOG_PATH, encoding="utf-8").read().strip()
        assert "tool=run_shell" in line
        assert '"command": "ls -la"' in line
        assert "rc=0" in line

    def test_args_serialized_utf8_readable(self):
        agent.log_tool("run_shell", {"command": "echo Привет"}, "ok", rc=0)
        line = open(agent.TOOL_LOG_PATH, encoding="utf-8").read()
        assert "Привет" in line  # ensure_ascii=False — читаемый русский текст


# --------------------------------------------------------------- save_history --

class TestSaveHistory:
    def test_writes_jsonl_utf8(self):
        msgs = [
            {"role": "system", "content": "системный промпт"},
            {"role": "user", "content": "Привет, мир"},
            {"role": "assistant", "content": "Здравствуйте!"},
        ]
        agent.save_history(msgs)
        lines = open(agent.HISTORY_PATH, encoding="utf-8").read().splitlines()
        parsed = [json.loads(l) for l in lines]
        assert parsed == msgs
        assert "Здравствуйте!" in lines[2]  # без \uXXXX-экранирования

    def test_overwrites_previous_session(self):
        agent.save_history([{"role": "user", "content": "старое"}])
        agent.save_history([{"role": "user", "content": "новое"}])
        lines = open(agent.HISTORY_PATH, encoding="utf-8").read().splitlines()
        # файл перезаписывается целиком: одна строка — только новая сессия
        assert len(lines) == 1
        assert json.loads(lines[0]) == {"role": "user", "content": "новое"}


# ------------------------------------------------------------------ run_shell --

class TestRunShell:
    def _result(self, command, timeout=10):
        raw = agent.run_shell(command, timeout)
        return json.loads(raw), raw

    def test_stdout_returncode(self):
        res, _ = self._result("echo hello")
        assert res["stdout"].strip() == "hello"
        assert res["stderr"] == ""
        assert res["returncode"] == 0

    def test_stderr_and_nonzero_rc(self):
        res, _ = self._result("echo oops >&2; exit 3")
        assert res["stderr"].strip() == "oops"
        assert res["returncode"] == 3

    def test_full_shell_features(self):
        # pipes, перенаправление, переменные — полноценный bash
        res, _ = self._result('X="world"; echo "hello, $X" | tr a-z A-Z')
        assert "HELLO, WORLD" in res["stdout"]
        assert res["returncode"] == 0

    def test_full_shell_features_utf8_russian(self):
        # русские строки проходят через shell и возвращаются читаемыми (UTF-8)
        res, _ = self._result('X=мир; echo "привет, $X"')
        assert res["stdout"].strip() == "привет, мир"
        assert res["returncode"] == 0

    def test_timeout(self):
        res, _ = self._result("sleep 5", timeout=1)
        assert "error" in res
        assert "Таймаут" in res["error"]

    def test_tool_usage_log_written(self):
        _, raw = self._result("echo logged")
        log = open(agent.TOOL_LOG_PATH, encoding="utf-8").read()
        assert "tool=run_shell" in log
        assert "echo logged" in log
        assert "rc=0" in log

    def test_timeout_logged(self):
        self._result("sleep 5", timeout=1)
        log = open(agent.TOOL_LOG_PATH, encoding="utf-8").read()
        assert "rc=timeout" in log


# ------------------------------------------------------------ prompts & tool --

class TestSystemPromptAndToolSchema:
    def test_system_prompt_forbids_unreadable_utf8(self):
        sp = agent.SYSTEM_PROMPT
        assert "UTF-8" in sp
        assert "нечитаем" in sp  # «нечитаемые символы» — явный запрет

    def test_run_shell_tool_schema(self):
        t = agent.RUN_SHELL_TOOL
        assert t["type"] == "function"
        fn = t["function"]
        assert fn["name"] == "run_shell"
        assert "command" in fn["parameters"]["required"]
        assert "command" in fn["parameters"]["properties"]


# ---------------------------------------------------------------------- main --

def make_response(content=None, tool_calls=None):
    """Конструирует объект ответа в стиле openai-пакета."""
    from types import SimpleNamespace

    msg = SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in tool_calls
        ]
        if tool_calls
        else None,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class FakeClient:
    """Мок OpenAI-клиента: возвращает заранее заготовленные ответы."""

    def __init__(self, responses):
        self.calls = []
        seq = iter(responses)

        def create(**kw):
            self.calls.append(kw)
            return next(seq)

        completions = SimpleNamespace(create=create)
        self.chat = SimpleNamespace(completions=completions)


def make_input(lines):
    """input() для monkeypatch: отдаёт строки из `lines`, затем 'exit'.

    Добавленный хвостовой 'exit' гарантирует штатное завершение main()
    даже если мок-ответы закончились раньше (StopIteration -> EOF -> break).
    """
    it = iter(list(lines) + ["exit"])

    def _input(*_):
        try:
            return next(it)
        except StopIteration:
            raise EOFError  # main() трактует это как Ctrl+C и корректно выходит
    return _input


class TestMainIntegration:
    def test_plain_chat_flow(self, monkeypatch):
        fake = FakeClient([make_response(content="Привет!")])
        monkeypatch.setattr(agent, "OpenAI", lambda **kw: fake)
        monkeypatch.setattr(agent, "load_config", lambda p: {
            "api_key": "sk", "base_url": "http://x/v1", "model": "m"})
        monkeypatch.setattr("builtins.input", make_input(["здравствуйте"]))

        agent.main()  # ответ модели один, затем 'exit' завершает сессию штатно

        # tools (описание function_call) передаётся в каждом запросе
        assert fake.calls[0]["tools"] == [agent.RUN_SHELL_TOOL]
        assert fake.calls[0]["messages"][0]["content"] == agent.SYSTEM_PROMPT
        # лог API: текст модели записан
        api_log = open(agent.API_LOG_PATH, encoding="utf-8").read()
        assert "user: здравствуйте" in api_log
        assert "model: Привет!" in api_log
        # история сохранена в отдельный файл
        hist = [json.loads(l) for l in open(agent.HISTORY_PATH, encoding="utf-8")]
        assert hist[-1] == {"role": "assistant", "content": "Привет!"}

    def test_tool_call_round_trip(self, monkeypatch):
        """Модель вызывает run_shell -> результат уходит обратно как role='tool'."""
        tc = [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "run_shell", "arguments": json.dumps({"command": "echo тост"})},
        }]
        fake = FakeClient([
            make_response(tool_calls=tc),
            make_response(content="Команда выполнена."),
        ])
        monkeypatch.setattr(agent, "OpenAI", lambda **kw: fake)
        monkeypatch.setattr(agent, "load_config", lambda p: {
            "api_key": "sk", "base_url": "http://x/v1", "model": "m"})
        monkeypatch.setattr("builtins.input", make_input(["выполни echo тост"]))

        agent.main()  # tool-цикл завершается ответом модели, затем 'exit'

        assert len(fake.calls) == 2  # второй запрос — уже с результатом инструмента
        second_msgs = fake.calls[1]["messages"]
        tool_msg = next(m for m in reversed(second_msgs) if m["role"] == "tool")
        asst = second_msgs[second_msgs.index(tool_msg) - 1]
        assert asst["role"] == "assistant" and asst["tool_calls"][0]["id"] == "call_1"
        assert tool_msg["tool_call_id"] == "call_1"
        result = json.loads(tool_msg["content"])
        assert result["stdout"].strip() == "тост"
        assert result["returncode"] == 0
        # журнал использования TOOL заполнен реальным вызовом
        tool_log = open(agent.TOOL_LOG_PATH, encoding="utf-8").read()
        assert "tool=run_shell" in tool_log and "echo тост" in tool_log

    def test_unknown_tool_handled(self, monkeypatch):
        tc = [{
            "id": "call_x",
            "type": "function",
            "function": {"name": "no_such_tool", "arguments": "{}"},
        }]
        fake = FakeClient([
            make_response(tool_calls=tc),
            make_response(content="Инструмент недоступен."),
        ])
        monkeypatch.setattr(agent, "OpenAI", lambda **kw: fake)
        monkeypatch.setattr(agent, "load_config", lambda p: {
            "api_key": "sk", "base_url": "http://x/v1", "model": "m"})
        # после финального текстового ответа main() вернётся к input():
        # хвостовой 'exit' из make_input штатно завершает сессию.
        monkeypatch.setattr("builtins.input", make_input(["сделай что-то"]))

        agent.main()

        tool_msgs = [m for m in fake.calls[1]["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        tool_msg = tool_msgs[0]
        assert tool_msg["tool_call_id"] == "call_x"
        assert "Неизвестный инструмент" in json.loads(tool_msg["content"])["error"]

    def test_clear_command_resets_history(self, monkeypatch):
        fake = FakeClient([make_response(content="Ответ"), make_response(content="Ответ2")])
        monkeypatch.setattr(agent, "OpenAI", lambda **kw: fake)
        monkeypatch.setattr(agent, "load_config", lambda p: {
            "api_key": "sk", "base_url": "http://x/v1", "model": "m"})
        inputs = ["первый", "clear", "второй"]
        monkeypatch.setattr("builtins.input", make_input(inputs))

        agent.main()  # два ответа модели consumed по одному на ход, затем 'exit'

        hist = [json.loads(l) for l in open(agent.HISTORY_PATH, encoding="utf-8")]
        roles_content = [m.get("content") for m in hist]
        assert "первый" not in roles_content  # до clear стёрта
        assert "второй" in roles_content

    def test_bad_config_exits(self, monkeypatch):
        def raise_(p):
            raise agent.ConfigError("нет файла")
        monkeypatch.setattr(agent, "load_config", raise_)
        with pytest.raises(SystemExit) as exc:
            agent.main()
        assert exc.value.code == 1


# --------------------------------------------------------------- документация --

class TestDocumentation:
    """Код должен быть максимально документированным (требование задачи)."""

    def test_module_docstring_covers_key_requirements(self):
        """Модульный docstring описывает назначение, возможности и запуск."""
        doc = agent.__doc__ or ""
        for keyword in ("НАЗНАЧЕНИЕ", "ВОЗМОЖНОСТИ", "config.yaml", "run_shell",
                        "api.log", "tool_usage.log", "chat_history.jsonl",
                        "UTF-8", "20 символов", "БЫСТРЫЙ СТАРТ"):
            assert keyword in doc, f"в модульном docstring нет раздела/слова: {keyword!r}"

    def test_all_public_functions_have_docstrings(self):
        """Каждая публичная функция модуля имеет docstring с Args/Returns или Raises."""
        funcs = [agent.truncate, agent.load_config, agent.validate_config,
                 agent.log_api, agent.log_tool, agent.save_history,
                 agent.run_shell, agent.main]
        for fn in funcs:
            doc = fn.__doc__ or ""
            assert len(doc.split()) >= 5, f"{fn.__name__}: docstring слишком короткий"

    def test_todo_sections_for_model_switch_exist(self):
        """TODO-разделы по выбору модели во время сессии присутствуют в коде."""
        src = open(agent.__file__, encoding="utf-8").read()
        # Планируемые команды и связанные аспекты — каждый помечен TODO(...):
        for marker in ("TODO(model-switch)", "TODO(model-list)",
                       "TODO(model-validate)", "TODO(model-history)",
                       "TODO(model-log)", "TODO(model-config-fallback)",
                       "TODO(tests)"):
            assert marker in src, f"в agent.py отсутствует раздел {marker}"
        # TODO-маркеры должны встречаться и в докстринге main(), и в инлайн-комментариях
        assert agent.main.__doc__.count("TODO(model-") >= 3
        assert "TODO(model-switch)" in src[src.index("def main"):]

    def test_inline_todos_near_change_points(self):
        """Инлайн TODO стоят рядом с точками будущих изменений (переменная model)."""
        src = open(agent.__file__, encoding="utf-8").read()
        idx_var = src.index('model = cfg.get("model", DEFAULT_MODEL)')
        window = src[max(0, idx_var - 300):idx_var]
        assert "TODO(model-switch)" in window, \
            "рядом с инициализацией текущей модели нет TODO о её смене в сессии"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
