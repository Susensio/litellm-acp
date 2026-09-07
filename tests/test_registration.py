import importlib

import litellm


def test_import_registers_acp_provider():
    import litellm_acp

    matches = [entry for entry in litellm.custom_provider_map if entry.get("provider") == "acp"]
    assert len(matches) == 1
    assert isinstance(matches[0]["custom_handler"], litellm_acp.ACPCustomLLM)


def test_reimport_does_not_duplicate_or_clobber_other_entries():
    import litellm_acp

    litellm.custom_provider_map = [
        {"provider": "someone-elses-provider", "custom_handler": object()}
    ]
    importlib.reload(litellm_acp)

    providers = [entry["provider"] for entry in litellm.custom_provider_map]
    assert providers.count("acp") == 1
    assert "someone-elses-provider" in providers
