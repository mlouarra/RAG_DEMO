from typing import Any

from fm_rag.ollama_client import OllamaChatClient


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {
            "message": {
                "role": "assistant",
                "content": '  {"action": "measure_evolution"}  ',
            }
        }


def test_chat_sends_expected_request(
    monkeypatch,
) -> None:
    captured_request: dict[str, Any] = {}

    def fake_post(
        url: str,
        *,
        json: dict[str, Any],
        timeout: int,
    ) -> FakeResponse:
        captured_request["url"] = url
        captured_request["json"] = json
        captured_request["timeout"] = timeout

        return FakeResponse()

    monkeypatch.setattr(
        "fm_rag.ollama_client.requests.post",
        fake_post,
    )

    client = OllamaChatClient(
        base_url="http://localhost:11434/",
        model="mistral:7b",
        timeout=30,
        temperature=0.0,
    )

    response = client.chat(
        messages=[
            {
                "role": "user",
                "content": " Analyse cette question. ",
            }
        ],
        response_format={
            "type": "object",
        },
    )

    assert response == '{"action": "measure_evolution"}'
    assert (
        captured_request["url"]
        == "http://localhost:11434/api/chat"
    )
    assert captured_request["timeout"] == 30
    assert captured_request["json"] == {
        "model": "mistral:7b",
        "messages": [
            {
                "role": "user",
                "content": "Analyse cette question.",
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
        "format": {
            "type": "object",
        },
    }