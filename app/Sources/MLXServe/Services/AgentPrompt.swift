import Foundation

enum AgentPrompt {

    static let systemPrompt = "You are a helpful macOS assistant. Use tools for tasks. Answer directly when no tools needed. For web search use webSearch tool."

    /// OpenAI-format tool definitions — kept minimal to reduce prompt tokens for small models.
    static let toolDefinitions: [[String: Any]] = [
        [
            "type": "function",
            "function": [
                "name": "shell",
                "description": "Run a command",
                "parameters": [
                    "type": "object",
                    "properties": ["command": ["type": "string"]],
                    "required": ["command"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "writeFile",
                "description": "Write a file",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"],
                        "content": ["type": "string"]
                    ],
                    "required": ["path", "content"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "readFile",
                "description": "Read a file",
                "parameters": [
                    "type": "object",
                    "properties": ["path": ["type": "string"]],
                    "required": ["path"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "editFile",
                "description": "Find and replace in file",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "path": ["type": "string"],
                        "find": ["type": "string"],
                        "replace": ["type": "string"]
                    ],
                    "required": ["path", "find", "replace"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "searchFiles",
                "description": "Grep for pattern",
                "parameters": [
                    "type": "object",
                    "properties": ["pattern": ["type": "string"]],
                    "required": ["pattern"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "browse",
                "description": "Browse a URL",
                "parameters": [
                    "type": "object",
                    "properties": [
                        "action": ["type": "string"],
                        "url": ["type": "string"]
                    ],
                    "required": ["action"]
                ]
            ] as [String: Any]
        ],
        [
            "type": "function",
            "function": [
                "name": "webSearch",
                "description": "Search the web",
                "parameters": [
                    "type": "object",
                    "properties": ["query": ["type": "string"]],
                    "required": ["query"]
                ]
            ] as [String: Any]
        ],
    ]
}
