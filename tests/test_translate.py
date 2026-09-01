from unittest.mock import MagicMock

from litellm_acp.translate import apply_response_text, messages_to_prompt


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


def test_apply_response_text_populates_model_response():
    message = MagicMock(content=None, role=None)
    choice = MagicMock(message=message, finish_reason=None)
    model_response = MagicMock(choices=[choice], model=None)

    result = apply_response_text(model_response, "the reply", model="acp/claude")

    assert result is model_response
    assert message.content == "the reply"
    assert message.role == "assistant"
    assert choice.finish_reason == "stop"
    assert model_response.model == "acp/claude"
