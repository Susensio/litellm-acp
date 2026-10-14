import pytest
from acp.schema import Cost, Usage
from litellm.types.utils import ModelResponse

from litellm_acp.translate import fill_model_response, messages_to_prompt, stream_chunk
from litellm_acp.turn import TextDelta, TurnEnd


def test_single_user_message_sent_verbatim():
    blocks = messages_to_prompt([{"role": "user", "content": "hi there"}])
    assert len(blocks) == 1
    assert blocks[0].text == "hi there"


def test_multi_turn_history_is_labeled_and_flattened():
    blocks = messages_to_prompt(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ]
    )
    assert len(blocks) == 1
    text = blocks[0].text
    assert "[System]\nbe terse" in text
    assert "[User]\nhi" in text
    assert "[Assistant]\nhello" in text
    assert "[User]\nhow are you" in text
    # order is preserved
    assert text.index("be terse") < text.index("[User]\nhi") < text.index("hello")


def test_multimodal_content_list_is_flattened_to_text():
    blocks = messages_to_prompt(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {"type": "image_url", "image_url": {}},
                ],
            }
        ]
    )
    assert blocks[0].text == "look at this<image_url>"


def test_empty_messages_yields_empty_block():
    blocks = messages_to_prompt([])
    assert blocks[0].text == ""


def _end(stop_reason="end_turn", usage=None, cost=None) -> TurnEnd:
    return TurnEnd(usage=usage, cost=cost, stop_reason=stop_reason)


def test_fill_model_response_populates_reply_and_model():
    response = fill_model_response(ModelResponse(), "the reply", _end(), model="acp/claude")

    assert response.choices[0].message.content == "the reply"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason == "stop"
    assert response.model == "acp/claude"


def test_usage_adds_cache_counters_back_into_prompt_tokens():
    # ACP reports input net of cache; OpenAI's prompt_tokens includes it.
    usage = Usage(
        input_tokens=10,
        output_tokens=20,
        cached_read_tokens=300,
        cached_write_tokens=40,
        thought_tokens=5,
        total_tokens=370,
    )

    response = fill_model_response(ModelResponse(), "", _end(usage=usage), model="acp/x")

    assert response.usage.prompt_tokens == 350
    assert response.usage.completion_tokens == 20
    assert response.usage.total_tokens == 370
    assert response.usage.prompt_tokens_details.cached_tokens == 300
    assert response.usage.cache_creation_input_tokens == 40
    assert response.usage.completion_tokens_details.reasoning_tokens == 5


def test_only_usd_costs_become_the_api_equivalent_field():
    usd = fill_model_response(
        ModelResponse(), "", _end(cost=Cost(amount=0.5, currency="USD")), model="acp/x"
    )
    eur = fill_model_response(
        ModelResponse(), "", _end(cost=Cost(amount=0.5, currency="EUR")), model="acp/x"
    )

    assert usd._hidden_params["acp_api_equivalent_cost"] == 0.5
    assert "acp_api_equivalent_cost" not in eur._hidden_params


def test_text_delta_becomes_an_unfinished_chunk():
    chunk = stream_chunk(TextDelta("hi"))

    assert chunk["text"] == "hi"
    assert chunk["is_finished"] is False
    assert chunk["usage"] is None


def test_turn_end_finishes_the_stream_with_usage_and_api_cost():
    usage = Usage(input_tokens=1, output_tokens=2, total_tokens=3)
    chunk = stream_chunk(_end(usage=usage, cost=Cost(amount=0.25, currency="USD")))

    assert chunk["is_finished"] is True
    assert chunk["usage"]["prompt_tokens"] == 1
    assert chunk["usage"]["completion_tokens"] == 2
    assert chunk["provider_specific_fields"] == {"acp_api_equivalent_cost": 0.25}


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("max_turn_requests", "length"),
        ("refusal", "content_filter"),
        ("cancelled", "stop"),
    ],
)
def test_acp_stop_reasons_map_to_openai_finish_reasons(stop_reason, expected):
    assert stream_chunk(_end(stop_reason))["finish_reason"] == expected
    response = fill_model_response(ModelResponse(), "", _end(stop_reason), model="acp/x")
    assert response.choices[0].finish_reason == expected
