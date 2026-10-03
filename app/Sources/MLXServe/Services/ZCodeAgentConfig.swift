import Foundation

extension AgentConfigs {
    /// Twin of src/zcode_launch.zig; sparse per-model overrides suppress
    /// upstream model-name guesses and carry the serving endpoint's limits.
    static func zcodeProviderJSON(baseURL: String, model: String, budget: AgentBudget.Budget,
                                  entries: [AgentModelEntry]) -> String {
        let models = entries.isEmpty ? [AgentModelEntry(id: model, budget: budget, vision: false)] : entries
        let rules: [[String: Any]] = models.map { entry in
            ["providerId": "mlx", "modelId": entry.id, "config": [
                "enabled": true,
                "properties": [
                    "contextWindow": entry.budget.context, "requiresMfjsToolSchema": false,
                    "inputFormat": ["supportsText": true, "supportsImage": entry.vision,
                                    "supportsVideo": false, "supportsAudio": false, "supportsPdf": false],
                    "outputFormat": ["supportsText": true], "supportsToolCall": true,
                    "supportsJsonSchemaOutput": false, "supportsNativeWebSearch": false,
                    "supportsMidConversationSystem": false,
                ],
                "optionSpecs": [
                    "reasoningLevel": ["values": ["none", "low", "medium", "high"],
                                       "map": "{\"reasoning_effort\": reasoningLevel}"],
                    "maxOutputTokens": ["max": entry.budget.output,
                                        "map": "{\"max_tokens\": maxOutputTokens}"],
                ],
            ]]
        }
        let config: [String: Any] = ["schemaVersion": 1, "config": [
            "providerOrder": ["mlx"],
            "defaultModelSelection": ["providerId": "mlx", "modelId": model,
                                      "options": ["reasoningLevel": "medium"]],
            "providerConfigRules": ["providerRules": [[
                "providerId": "mlx", "providerName": "MLX Serve / Sushi", "enabled": true,
                "config": ["group": "standard-personal",
                           "access": ["type": "api-key", "apiKey": "mlx-serve"],
                           "api": ["type": "openai-chat-completions", "baseUrl": "\(baseURL)/v1"],
                           "personalModelIds": models.map(\.id)],
            ]]],
            "modelConfigRules": ["manualProviderModelRules": [], "providerModelRules": rules],
        ]]
        let data = try! JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        return String(decoding: data, as: UTF8.self)
    }

    static let zcodeExports = #"""
    export ZCODE_DATA_BASE_DIR="$HOME/.mlx-serve/zcode"
    export ZCODE_STORAGE_DIR="$HOME/.mlx-serve/zcode/storage"
    export ZCODE_PERSONAL_PROVIDER_CONFIG_FILE="$HOME/.mlx-serve/zcode/provider_config.json"
    if ! command -v zcode >/dev/null 2>&1; then echo "zcode is not installed: build or install ZCode (https://github.com/zai-org/ZCode)" >&2; exit 127; fi
    """#
}
