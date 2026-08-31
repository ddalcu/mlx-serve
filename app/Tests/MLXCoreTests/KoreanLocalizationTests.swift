import XCTest

final class KoreanLocalizationTests: XCTestCase {
    private var appRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private func koreanStrings() throws -> [String: String] {
        let url = appRoot
            .appendingPathComponent("Sources/MLXServe/Resources/ko.lproj/Localizable.strings")
        let data = try Data(contentsOf: url)
        var format = PropertyListSerialization.PropertyListFormat.openStep
        let value = try PropertyListSerialization.propertyList(
            from: data,
            options: [],
            format: &format
        )
        return try XCTUnwrap(value as? [String: String])
    }

    func testKoreanLocalizationExistsAndParses() throws {
        XCTAssertGreaterThan(try koreanStrings().count, 100)
    }

    func testCoreNavigationAndSafetyCopyAreTranslated() throws {
        let strings = try koreanStrings()
        let required = [
            "New Chat": "새 대화",
            "Models": "모델",
            "Settings": "설정",
            "Agents": "에이전트",
            "Tasks": "작업",
            "Ask me anything…": "무엇이든 물어보세요…",
            "Allow this tool call?": "이 도구 실행을 허용할까요?",
            "Deny": "거부",
            "Allow": "허용",
            "Delete": "삭제",
            "Cancel": "취소",
            "This can't be undone.": "이 작업은 되돌릴 수 없습니다.",
            "Restart Now": "지금 다시 시작",
            "Discard": "변경 취소",
            "Browse all models": "모든 모델 찾아보기",
            "Start Chatting": "채팅 시작",
        ]

        for (key, expected) in required {
            XCTAssertEqual(strings[key], expected, "Missing or incorrect Korean value for \(key)")
        }
    }

    func testImportantFormatPlaceholdersArePreserved() throws {
        let strings = try koreanStrings()
        let requiredPlaceholderKeys = [
            "%@ bytes",
            "%@ token%@",
            "(%@ tokens)",
            "Tool: %@",
            "GPU %@%%",
            "Mem %@%%",
        ]

        for key in requiredPlaceholderKeys {
            let translated = try XCTUnwrap(strings[key], "Missing Korean value for \(key)")
            XCTAssertEqual(placeholders(in: translated), placeholders(in: key), key)
        }
    }

    func testEveryTranslatedFormatStringPreservesItsPlaceholders() throws {
        for (key, translated) in try koreanStrings() where key.contains("%") {
            XCTAssertEqual(
                placeholders(in: translated),
                placeholders(in: key),
                "Format placeholders changed for \(key)"
            )
        }
    }

    private func placeholders(in value: String) -> [String] {
        let pattern = #"%(?:\d+\$)?(?:@|d|ld|lld|f|\.\d+f|%)"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let range = NSRange(value.startIndex..<value.endIndex, in: value)
        return regex.matches(in: value, range: range).compactMap {
            Range($0.range, in: value).map { String(value[$0]) }
        }
    }
}
