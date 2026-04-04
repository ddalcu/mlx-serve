import Foundation

enum AgentPrompt {

    static let systemPrompt = """
    You are an AI assistant with tool access. When the user asks you to perform a task that requires \
    tools, create a plan. If you need clarification, ask questions instead.

    Available tools:
    - shell: Run a shell command. Params: {"command": "..."}
    - readFile: Read a file. Params: {"path": "...", "startLine": "N", "endLine": "N"}
    - writeFile: Write to a file. Params: {"path": "...", "content": "..."}
    - editFile: Find and replace in a file. Params: {"path": "...", "find": "...", "replace": "..."}
    - searchFiles: Grep for a pattern. Params: {"pattern": "...", "path": "..."}
    - webFetch: Fetch a URL as text. Params: {"url": "..."}
    - browse: Control Safari. Params: {"action": "navigate|readText|click|getInfo|executeJS", "url": "...", "selector": "...", "script": "..."}
    - browseChrome: Control Chrome via CDP (needs --remote-debugging-port=9222). Params: {"action": "navigate|readText|click|evaluate", "url": "...", "selector": "...", "expression": "..."}
    - applescript: Run AppleScript. Params: {"source": "..."}

    To use tools, output a JSON plan inside <plan> tags:
    <plan>
    [
      {"tool": "shell", "description": "List project files", "parameters": {"command": "ls -la"}}
    ]
    </plan>

    Rules:
    - Each step needs: tool (string), description (string), parameters (object with string values).
    - Keep plans concise. The user reviews and approves before execution.
    - After execution, you receive results and should summarize what happened.
    """

    /// Parse a `<plan>...</plan>` block from model output into an AgentPlan.
    static func parsePlan(from text: String) -> AgentPlan? {
        guard let startRange = text.range(of: "<plan>"),
              let endRange = text.range(of: "</plan>") else { return nil }

        let jsonStr = String(text[startRange.upperBound..<endRange.lowerBound])
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Lenient: strip trailing commas before ] and }
        let cleaned = jsonStr
            .replacingOccurrences(of: ",\\s*]", with: "]", options: .regularExpression)
            .replacingOccurrences(of: ",\\s*}", with: "}", options: .regularExpression)

        guard let data = cleaned.data(using: .utf8),
              let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            return nil
        }

        var steps: [PlanStep] = []
        for item in array {
            guard let toolStr = item["tool"] as? String,
                  let tool = AgentToolKind(rawValue: toolStr),
                  let description = item["description"] as? String else { continue }

            let rawParams = (item["parameters"] as? [String: Any]) ?? [:]
            let params = rawParams.reduce(into: [String: String]()) { result, pair in
                result[pair.key] = "\(pair.value)"
            }
            steps.append(PlanStep(tool: tool, description: description, parameters: params))
        }

        return steps.isEmpty ? nil : AgentPlan(steps: steps)
    }

    /// Remove the `<plan>...</plan>` block from text, leaving surrounding content.
    static func stripPlanTag(from text: String) -> String {
        guard let startRange = text.range(of: "<plan>"),
              let endRange = text.range(of: "</plan>") else { return text }
        var result = text
        result.removeSubrange(startRange.lowerBound..<endRange.upperBound)
        return result.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Format step results as concise text for the model to summarize.
    static func formatResults(_ results: [StepResult], plan: AgentPlan) -> String {
        var parts: [String] = []
        for (i, step) in plan.steps.enumerated() {
            var line = "Step \(i + 1) [\(step.tool.rawValue)]: \(step.description)"
            let paramStr = step.parameters.map { "  \($0.key): \($0.value)" }.joined(separator: "\n")
            if !paramStr.isEmpty { line += "\n\(paramStr)" }

            if i < results.count {
                let r = results[i]
                line += "\n  Status: \(r.status.rawValue)"
                if !r.output.isEmpty {
                    line += "\n  Output:\n\(String(r.output.prefix(2000)))"
                }
                if let error = r.error, !error.isEmpty {
                    line += "\n  Error: \(error)"
                }
            }
            parts.append(line)
        }
        return parts.joined(separator: "\n\n")
    }
}
