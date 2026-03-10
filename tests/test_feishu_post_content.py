# Tests for Feishu post content extraction (rich text with images) and
# the optional event registration helper for SDK version compatibility.

from nanobot.channels.feishu import FeishuChannel, _extract_post_content


# Verify nested post wrapper (post -> zh_cn -> content) is correctly extracted
def test_extract_post_content_supports_post_wrapper_shape() -> None:
    payload = {
        "post": {
            "zh_cn": {
                "title": "日报",
                "content": [
                    [
                        {"tag": "text", "text": "完成"},
                        {"tag": "img", "image_key": "img_1"},
                    ]
                ],
            }
        }
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "日报 完成"
    assert image_keys == ["img_1"]


# Verify flat/direct post shape (title + content at top level) also works
def test_extract_post_content_keeps_direct_shape_behavior() -> None:
    payload = {
        "title": "Daily",
        "content": [
            [
                {"tag": "text", "text": "report"},
                {"tag": "img", "image_key": "img_a"},
                {"tag": "img", "image_key": "img_b"},
            ]
        ],
    }

    text, image_keys = _extract_post_content(payload)

    assert text == "Daily report"
    assert image_keys == ["img_a", "img_b"]


# Verify _register_optional_event is a no-op when the builder lacks the method
def test_register_optional_event_keeps_builder_when_method_missing() -> None:
    class Builder:
        pass

    builder = Builder()
    same = FeishuChannel._register_optional_event(builder, "missing", object())
    assert same is builder


# Verify _register_optional_event calls the method when it exists on the builder
def test_register_optional_event_calls_supported_method() -> None:
    called = []

    class Builder:
        def register_event(self, handler):
            called.append(handler)
            return self

    builder = Builder()
    handler = object()
    same = FeishuChannel._register_optional_event(builder, "register_event", handler)

    assert same is builder
    assert called == [handler]
